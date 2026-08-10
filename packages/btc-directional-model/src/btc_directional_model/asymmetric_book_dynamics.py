"""Causal Polymarket book dynamics for asymmetric entry evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np
import polars as pl

from .asymmetric_value_data import POLYMARKET_VALUE_FEATURES

BOOK_DYNAMICS_SCHEMA_VERSION = "btc-asymmetric-book-dynamics-v1"
BOOK_DYNAMICS_HORIZONS_SECONDS = (1, 5, 15)
BOOK_DYNAMICS_KEY_COLUMNS = (
    "market_id",
    "window_start",
    "observed_at",
    "seconds_elapsed",
)
SELECTED_SIDE_COLUMN = "selected_side"


@dataclass(frozen=True)
class BookDeltaBase:
    """One frozen current-minus-causal-lag transformation."""

    name: str
    orientation: Literal["selected", "direct"]
    yes_feature: str
    no_feature: str | None = None


BOOK_DELTA_BASES = (
    BookDeltaBase(
        name="selected_cost",
        orientation="selected",
        yes_feature="pm_yes_cost_per_share",
        no_feature="pm_no_cost_per_share",
    ),
    BookDeltaBase(
        name="opposite_cost",
        orientation="selected",
        yes_feature="pm_no_cost_per_share",
        no_feature="pm_yes_cost_per_share",
    ),
    BookDeltaBase(
        name="selected_vwap_slippage",
        orientation="selected",
        yes_feature="pm_yes_vwap_slippage",
        no_feature="pm_no_vwap_slippage",
    ),
    BookDeltaBase(
        name="opposite_vwap_slippage",
        orientation="selected",
        yes_feature="pm_no_vwap_slippage",
        no_feature="pm_yes_vwap_slippage",
    ),
    BookDeltaBase(
        name="selected_depth_log",
        orientation="selected",
        yes_feature="pm_yes_depth_log",
        no_feature="pm_no_depth_log",
    ),
    BookDeltaBase(
        name="opposite_depth_log",
        orientation="selected",
        yes_feature="pm_no_depth_log",
        no_feature="pm_yes_depth_log",
    ),
    BookDeltaBase(
        name="cost_overround",
        orientation="direct",
        yes_feature="pm_cost_overround",
    ),
    BookDeltaBase(
        name="depth_imbalance",
        orientation="direct",
        yes_feature="pm_depth_imbalance",
    ),
)

BOOK_DYNAMICS_DELTA_FEATURES = tuple(
    f"pm_{base.name}_delta_{horizon}s"
    for horizon in BOOK_DYNAMICS_HORIZONS_SECONDS
    for base in BOOK_DELTA_BASES
)
BOOK_DYNAMICS_MATURITY_FEATURES = tuple(
    f"pm_book_horizon_{horizon}s_mature"
    for horizon in BOOK_DYNAMICS_HORIZONS_SECONDS
)
BOOK_DYNAMICS_FEATURES = (
    *POLYMARKET_VALUE_FEATURES,
    *BOOK_DYNAMICS_DELTA_FEATURES,
    *BOOK_DYNAMICS_MATURITY_FEATURES,
)
EXPECTED_BOOK_DYNAMICS_FEATURE_COUNT = 40


def attach_causal_book_dynamics(
    frame: pl.DataFrame,
    *,
    maximum_book_age_seconds: float = 2.0,
) -> pl.DataFrame:
    """Attach exact-key 1/5/15-second book changes without future or row-shift joins.

    Each delta is oriented to the side selected on the current row. A horizon is
    mature only when its exact same-market, same-window timestamp exists in the
    supplied strict frame. Immature or source-unavailable horizons receive zero
    deltas and an explicit false maturity flag.
    """

    _validate_source_frame(frame, maximum_book_age_seconds=maximum_book_age_seconds)
    dynamic_overlap = sorted(
        {*BOOK_DYNAMICS_DELTA_FEATURES, *BOOK_DYNAMICS_MATURITY_FEATURES}
        & set(frame.columns)
    )
    if dynamic_overlap:
        raise ValueError(
            "book-dynamics source already contains derived columns: "
            + ", ".join(dynamic_overlap)
        )

    original_columns = tuple(frame.columns)
    enriched = frame.with_row_index("__book_dynamics_row")
    lag_features = tuple(
        dict.fromkeys(
            feature
            for base in BOOK_DELTA_BASES
            for feature in (base.yes_feature, base.no_feature)
            if feature is not None
        )
    )

    for horizon in BOOK_DYNAMICS_HORIZONS_SECONDS:
        lag_observed = f"__pm_lag_observed_at_{horizon}s"
        lag_second = f"__pm_lag_seconds_elapsed_{horizon}s"
        maturity = f"pm_book_horizon_{horizon}s_mature"
        lag = frame.select(
            "market_id",
            "window_start",
            (pl.col("observed_at") + pl.duration(seconds=horizon)).alias("observed_at"),
            (pl.col("seconds_elapsed") + horizon).alias("seconds_elapsed"),
            pl.col("observed_at").alias(lag_observed),
            pl.col("seconds_elapsed").alias(lag_second),
            *(pl.col(feature).alias(_lag_name(feature, horizon)) for feature in lag_features),
        )
        enriched = enriched.join(
            lag,
            on=list(BOOK_DYNAMICS_KEY_COLUMNS),
            how="left",
            validate="1:1",
            maintain_order="left",
        ).with_columns(
            (
                (pl.col("seconds_elapsed") > horizon)
                & pl.col(lag_observed).is_not_null()
            ).alias(maturity)
        )

        causal_violations = enriched.filter(
            pl.col(maturity)
            & (
                (pl.col(lag_observed) >= pl.col("observed_at"))
                | (
                    (pl.col("observed_at") - pl.col(lag_observed)).dt.total_microseconds()
                    != horizon * 1_000_000
                )
                | (pl.col(lag_second) != pl.col("seconds_elapsed") - horizon)
            )
        )
        if causal_violations.height:
            raise RuntimeError(f"{horizon}s book-dynamics lag violated causal key alignment")

        deltas: list[pl.Expr] = []
        for base in BOOK_DELTA_BASES:
            current, prior = _oriented_values(base, horizon)
            deltas.append(
                pl.when(pl.col(maturity))
                .then(current - prior)
                .otherwise(0.0)
                .cast(pl.Float64)
                .alias(f"pm_{base.name}_delta_{horizon}s")
            )
        enriched = enriched.with_columns(*deltas)

    hidden = [column for column in enriched.columns if column.startswith("__pm_lag_")]
    enriched = (
        enriched.sort("__book_dynamics_row")
        .drop("__book_dynamics_row", *hidden)
        .select(
            *original_columns,
            *BOOK_DYNAMICS_DELTA_FEATURES,
            *BOOK_DYNAMICS_MATURITY_FEATURES,
        )
    )
    _validate_enriched_frame(enriched)
    if len(BOOK_DYNAMICS_FEATURES) != EXPECTED_BOOK_DYNAMICS_FEATURE_COUNT:
        raise RuntimeError("book-dynamics feature count changed")
    return enriched


def book_dynamics_schema_sha256() -> str:
    """Return the deterministic identity of the frozen feature transformation."""

    contract = {
        "schema_version": BOOK_DYNAMICS_SCHEMA_VERSION,
        "key_columns": list(BOOK_DYNAMICS_KEY_COLUMNS),
        "selected_side_column": SELECTED_SIDE_COLUMN,
        "horizons_seconds": list(BOOK_DYNAMICS_HORIZONS_SECONDS),
        "current_features": list(POLYMARKET_VALUE_FEATURES),
        "delta_bases": [
            {
                "name": base.name,
                "orientation": base.orientation,
                "yes_feature": base.yes_feature,
                "no_feature": base.no_feature,
            }
            for base in BOOK_DELTA_BASES
        ],
        "delta_features": list(BOOK_DYNAMICS_DELTA_FEATURES),
        "maturity_features": list(BOOK_DYNAMICS_MATURITY_FEATURES),
        "unavailable_horizon_value": 0.0,
        "join_contract": "exact_market_window_observed_at_seconds_elapsed",
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def book_dynamics_key_sha256(frame: pl.DataFrame) -> str:
    """Hash order-invariant exact decision keys."""

    _require_columns(frame, BOOK_DYNAMICS_KEY_COLUMNS, label="book-dynamics key digest")
    _validate_decision_keys(frame)
    ordered = frame.select(*BOOK_DYNAMICS_KEY_COLUMNS).sort(*BOOK_DYNAMICS_KEY_COLUMNS)
    digest = hashlib.sha256(b"btc-asymmetric-book-dynamics-key-v1\n")
    for row in ordered.iter_rows():
        for value in row:
            _update_scalar_digest(digest, value)
    return digest.hexdigest()


def book_dynamics_content_sha256(frame: pl.DataFrame) -> str:
    """Hash keys, side orientation, and the complete 40-feature contract."""

    required = (*BOOK_DYNAMICS_KEY_COLUMNS, SELECTED_SIDE_COLUMN, *BOOK_DYNAMICS_FEATURES)
    _require_columns(frame, required, label="book-dynamics content digest")
    _validate_enriched_frame(frame)
    ordered = frame.sort(*BOOK_DYNAMICS_KEY_COLUMNS)
    digest = hashlib.sha256(b"btc-asymmetric-book-dynamics-content-v1\n")
    digest.update(book_dynamics_schema_sha256().encode())
    digest.update(book_dynamics_key_sha256(ordered).encode())
    for value in ordered[SELECTED_SIDE_COLUMN].to_list():
        _update_scalar_digest(digest, value)
    for feature in (*POLYMARKET_VALUE_FEATURES, *BOOK_DYNAMICS_DELTA_FEATURES):
        digest.update(feature.encode())
        values = ordered[feature].cast(pl.Float64).to_numpy().astype("<f8", copy=False)
        values = np.where(values == 0.0, 0.0, values).astype("<f8", copy=False)
        digest.update(values.tobytes())
    for feature in BOOK_DYNAMICS_MATURITY_FEATURES:
        digest.update(feature.encode())
        digest.update(ordered[feature].cast(pl.UInt8).to_numpy().tobytes())
    return digest.hexdigest()


def book_dynamics_evidence(frame: pl.DataFrame) -> dict[str, Any]:
    """Summarize immutable feature identity and exact-horizon availability."""

    _require_columns(
        frame,
        (*BOOK_DYNAMICS_MATURITY_FEATURES, *BOOK_DYNAMICS_FEATURES),
        label="book-dynamics evidence",
    )
    return {
        "schema_version": BOOK_DYNAMICS_SCHEMA_VERSION,
        "schema_sha256": book_dynamics_schema_sha256(),
        "key_sha256": book_dynamics_key_sha256(frame),
        "content_sha256": book_dynamics_content_sha256(frame),
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "feature_count": len(BOOK_DYNAMICS_FEATURES),
        "feature_names": list(BOOK_DYNAMICS_FEATURES),
        "horizon_maturity": {
            f"{horizon}s": _horizon_maturity_evidence(frame, horizon)
            for horizon in BOOK_DYNAMICS_HORIZONS_SECONDS
        },
    }


def _validate_source_frame(
    frame: pl.DataFrame,
    *,
    maximum_book_age_seconds: float,
) -> None:
    if frame.is_empty():
        raise ValueError("book-dynamics source frame is empty")
    if not np.isfinite(maximum_book_age_seconds) or maximum_book_age_seconds <= 0:
        raise ValueError("maximum book age must be positive and finite")
    required = (*BOOK_DYNAMICS_KEY_COLUMNS, SELECTED_SIDE_COLUMN, *POLYMARKET_VALUE_FEATURES)
    _require_columns(frame, required, label="book-dynamics source")
    _validate_decision_keys(frame)
    _validate_selected_sides(frame)
    _validate_current_features(frame)
    age_invalid = frame.filter(
        ~pl.col("pm_yes_book_age_seconds").is_between(
            0.0, maximum_book_age_seconds, closed="both"
        )
        | ~pl.col("pm_no_book_age_seconds").is_between(
            0.0, maximum_book_age_seconds, closed="both"
        )
    )
    if age_invalid.height:
        raise ValueError("book-dynamics source contains non-causal or stale books")
    _validate_optional_receipt_times(frame, maximum_book_age_seconds)


def _horizon_maturity_evidence(frame: pl.DataFrame, horizon: int) -> dict[str, int | float | None]:
    maturity = f"pm_book_horizon_{horizon}s_mature"
    available_rows = int(frame[maturity].sum())
    causally_mature = frame.filter(pl.col("seconds_elapsed") > horizon)
    causally_mature_rows = causally_mature.height
    conditional_available_rows = int(causally_mature[maturity].sum())
    return {
        "available_rows": available_rows,
        "raw_rate": available_rows / frame.height,
        "causally_mature_rows": causally_mature_rows,
        "conditional_available_rows": conditional_available_rows,
        "conditional_available_rate": (
            conditional_available_rows / causally_mature_rows
            if causally_mature_rows
            else None
        ),
    }


def _validate_decision_keys(frame: pl.DataFrame) -> None:
    if frame.select(*BOOK_DYNAMICS_KEY_COLUMNS).is_duplicated().any():
        raise ValueError("book-dynamics source contains duplicate decision keys")
    for timestamp in ("window_start", "observed_at"):
        dtype = frame.schema[timestamp]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
            raise TypeError(f"{timestamp} must be a UTC Datetime")
    if not frame.schema["seconds_elapsed"].is_integer():
        raise TypeError("seconds_elapsed must be an integer")
    if frame.filter(pl.col("seconds_elapsed") < 0).height:
        raise ValueError("seconds_elapsed cannot precede market open")
    misaligned = frame.filter(
        (pl.col("observed_at") - pl.col("window_start")).dt.total_microseconds()
        != pl.col("seconds_elapsed").cast(pl.Int64) * 1_000_000
    )
    if misaligned.height:
        raise ValueError("book-dynamics timestamps do not match market-relative seconds")


def _validate_selected_sides(frame: pl.DataFrame) -> None:
    sides = set(frame[SELECTED_SIDE_COLUMN].unique().to_list())
    if not sides or not sides.issubset({"YES", "NO"}):
        raise ValueError("selected_side must contain only YES or NO")


def _validate_current_features(frame: pl.DataFrame) -> None:
    numeric_invalid = frame.filter(
        ~pl.all_horizontal(
            pl.col(feature).is_not_null()
            & pl.col(feature).cast(pl.Float64).is_finite()
            for feature in POLYMARKET_VALUE_FEATURES
        )
    )
    if numeric_invalid.height:
        raise ValueError("book-dynamics source contains non-finite PM features")


def _validate_optional_receipt_times(
    frame: pl.DataFrame,
    maximum_book_age_seconds: float,
) -> None:
    receipt_columns = ("yes_received_at", "no_received_at")
    present = [column in frame.columns for column in receipt_columns]
    if any(present) and not all(present):
        raise ValueError("book-dynamics source must provide both receipt timestamps or neither")
    if not all(present):
        return
    for side, receipt, age in (
        ("YES", "yes_received_at", "pm_yes_book_age_seconds"),
        ("NO", "no_received_at", "pm_no_book_age_seconds"),
    ):
        dtype = frame.schema[receipt]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
            raise TypeError(f"{receipt} must be a UTC Datetime")
        observed_age = (
            (pl.col("observed_at") - pl.col(receipt)).dt.total_milliseconds().cast(pl.Float64)
            / 1_000.0
        )
        violations = frame.filter(
            pl.col(receipt).is_null()
            | (pl.col(receipt) > pl.col("observed_at"))
            | (observed_age > maximum_book_age_seconds)
            | ((observed_age - pl.col(age)).abs() > 1e-6)
        )
        if violations.height:
            raise ValueError(f"{side} receipt timestamp violates causal book age")


def _validate_enriched_frame(frame: pl.DataFrame) -> None:
    required = (
        SELECTED_SIDE_COLUMN,
        *POLYMARKET_VALUE_FEATURES,
        *BOOK_DYNAMICS_DELTA_FEATURES,
        *BOOK_DYNAMICS_MATURITY_FEATURES,
    )
    _require_columns(frame, required, label="book-dynamics output")
    _validate_selected_sides(frame)
    _validate_current_features(frame)
    invalid_delta = frame.filter(
        ~pl.all_horizontal(
            pl.col(feature).is_not_null() & pl.col(feature).is_finite()
            for feature in BOOK_DYNAMICS_DELTA_FEATURES
        )
    )
    if invalid_delta.height:
        raise RuntimeError("book-dynamics output contains non-finite deltas")
    for horizon in BOOK_DYNAMICS_HORIZONS_SECONDS:
        maturity = f"pm_book_horizon_{horizon}s_mature"
        if frame.schema[maturity] != pl.Boolean or frame[maturity].null_count():
            raise RuntimeError(f"{horizon}s book maturity flag is not complete boolean evidence")
        immature = ~pl.col(maturity)
        delta_features = [
            f"pm_{base.name}_delta_{horizon}s" for base in BOOK_DELTA_BASES
        ]
        if frame.filter(
            immature & pl.any_horizontal(pl.col(feature) != 0.0 for feature in delta_features)
        ).height:
            raise RuntimeError(f"{horizon}s immature book deltas are not neutral zero")


def _oriented_values(base: BookDeltaBase, horizon: int) -> tuple[pl.Expr, pl.Expr]:
    if base.orientation == "direct":
        return pl.col(base.yes_feature), pl.col(_lag_name(base.yes_feature, horizon))
    if base.no_feature is None:
        raise RuntimeError(f"selected book delta {base.name} has no NO feature")
    is_yes = pl.col(SELECTED_SIDE_COLUMN) == "YES"
    current = pl.when(is_yes).then(pl.col(base.yes_feature)).otherwise(pl.col(base.no_feature))
    prior = (
        pl.when(is_yes)
        .then(pl.col(_lag_name(base.yes_feature, horizon)))
        .otherwise(pl.col(_lag_name(base.no_feature, horizon)))
    )
    return current, prior


def _lag_name(feature: str, horizon: int) -> str:
    return f"__pm_lag_{horizon}s_{feature}"


def _require_columns(frame: pl.DataFrame, required: tuple[str, ...], *, label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: " + ", ".join(missing))


def _update_scalar_digest(digest: Any, value: object) -> None:
    if isinstance(value, datetime):
        rendered = value.isoformat()
    else:
        rendered = str(value)
    encoded = rendered.encode()
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
