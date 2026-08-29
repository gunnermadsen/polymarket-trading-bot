"""Market-block conformal risk calibration for a frozen TWAP predictor."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

CONTROL = "frozen_probability_control"
GLOBAL = "global_market_block_conformal"
DIRECTION_PRICE = "direction_price_block_conformal"
DIRECTION_TIME_PRICE = "direction_time_price_block_conformal"
CANDIDATES = (CONTROL, GLOBAL, DIRECTION_PRICE, DIRECTION_TIME_PRICE)
CHECKPOINT_SECONDS = tuple(range(30, 121, 5))
VWAP_QUANTITIES = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200)
ARTIFACT_SCHEMA_VERSION = "btc-twap-conformal-risk-artifact-v1"


@dataclass(frozen=True)
class ConformalCell:
    support_markets: int
    probability_residual_quantile: float
    margin_residual_quantile_bps: float


@dataclass(frozen=True)
class ConformalArtifact:
    schema_version: str
    candidate: str
    alpha: float
    minimum_cell_markets: int
    calibration_start: str
    calibration_end: str
    source_identity: str
    prediction_ledger_sha256: str
    feature_registry_sha256: str
    upstream_artifact_sha256: str
    cells: dict[str, dict[str, ConformalCell]]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConformalArtifact:
        values = dict(payload)
        values["cells"] = {
            level: {key: ConformalCell(**cell) for key, cell in rows.items()}
            for level, rows in values["cells"].items()
        }
        artifact = cls(**values)
        validate_artifact(artifact)
        return artifact


def validate_artifact(artifact: ConformalArtifact) -> None:
    if artifact.schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unexpected conformal artifact schema")
    if artifact.candidate not in CANDIDATES:
        raise ValueError("unexpected conformal candidate")
    if artifact.alpha != 0.10:
        raise ValueError("conformal alpha changed")
    if artifact.minimum_cell_markets != 100:
        raise ValueError("conditional-cell support changed")
    if artifact.candidate == CONTROL and artifact.cells:
        raise ValueError("frozen control cannot contain conformal cells")
    if artifact.candidate != CONTROL and "global" not in artifact.cells:
        raise ValueError("conformal artifact lacks global fallback")


def artifact_bytes(artifact: ConformalArtifact) -> bytes:
    return (
        json.dumps(artifact.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def write_artifact(path: Path, artifact: ConformalArtifact) -> str:
    validate_artifact(artifact)
    payload = artifact_bytes(artifact)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(payload)
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def load_artifact(path: Path) -> ConformalArtifact:
    return ConformalArtifact.from_dict(json.loads(path.read_text()))


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    scores = np.asarray(values, dtype=float)
    scores = scores[np.isfinite(scores)]
    if not len(scores):
        raise ValueError("conformal quantile requires finite support")
    rank = min(len(scores), math.ceil((len(scores) + 1) * (1.0 - alpha)))
    return float(np.partition(scores, rank - 1)[rank - 1])


def price_band(cost: float) -> str:
    if cost < 0.60:
        return "below_0.60"
    if cost < 0.70:
        return "0.60-0.70"
    if cost < 0.80:
        return "0.70-0.80"
    return "above_0.80"


def time_band(second: int) -> str:
    if second < 60:
        return "30-59"
    if second < 90:
        return "60-89"
    return "90-120"


def add_group_columns(frame: pl.DataFrame) -> pl.DataFrame:
    probability = frame["probability_up"].to_numpy().astype(float)
    predicted_up = probability >= 0.5
    up_cost = frame["up_ask_vwap_5"].fill_null(float("nan")).to_numpy().astype(float)
    down_cost = frame["down_ask_vwap_5"].fill_null(float("nan")).to_numpy().astype(float)
    selected_cost = np.where(predicted_up, up_cost, down_cost)
    seconds = frame["seconds_elapsed"].to_numpy().astype(int)
    directions = np.where(predicted_up, "UP", "DOWN")
    price_bands = np.array(
        [price_band(value) if math.isfinite(value) else "unavailable" for value in selected_cost],
        dtype=object,
    )
    time_bands = np.array([time_band(int(value)) for value in seconds], dtype=object)
    return frame.with_columns(
        pl.Series("predicted_up", predicted_up),
        pl.Series("selected_probability", np.where(predicted_up, probability, 1.0 - probability)),
        pl.Series("selected_cost_5", selected_cost),
        pl.Series("direction_band", directions),
        pl.Series("price_band", price_bands),
        pl.Series("time_band", time_bands),
    ).with_columns(
        (pl.col("direction_band") + pl.lit("|") + pl.col("price_band")).alias(
            "direction_price_group"
        ),
        (
            pl.col("direction_band")
            + pl.lit("|")
            + pl.col("time_band")
            + pl.lit("|")
            + pl.col("price_band")
        ).alias("direction_time_price_group"),
    )


def fit_conformal_artifact(
    calibration: pl.DataFrame,
    candidate: str,
    *,
    alpha: float,
    minimum_cell_markets: int,
    calibration_start: str,
    calibration_end: str,
    source_identity: str,
    prediction_ledger_sha256: str,
    feature_registry_sha256: str,
    upstream_artifact_sha256: str,
) -> tuple[ConformalArtifact, pl.DataFrame]:
    if candidate not in CANDIDATES:
        raise ValueError(candidate)
    if alpha != 0.10 or minimum_cell_markets != 100:
        raise ValueError("frozen conformal contract changed")
    if candidate == CONTROL:
        return (
            ConformalArtifact(
                ARTIFACT_SCHEMA_VERSION,
                candidate,
                alpha,
                minimum_cell_markets,
                calibration_start,
                calibration_end,
                source_identity,
                prediction_ledger_sha256,
                feature_registry_sha256,
                upstream_artifact_sha256,
                {},
            ),
            pl.DataFrame(),
        )
    grouped = add_group_columns(calibration)
    eligible = grouped.filter(
        pl.col("book_valid")
        & pl.col("selected_cost_5").is_finite()
        & pl.col("target_margin_bps").is_finite()
    ).with_columns(
        (
            pl.col("selected_probability")
            - (pl.col("predicted_up") == pl.col("label_up")).cast(pl.Float64)
        ).alias("probability_residual"),
        pl.max_horizontal(
            pl.col("margin_p05_bps") - pl.col("target_margin_bps"),
            pl.col("target_margin_bps") - pl.col("margin_p95_bps"),
            pl.lit(0.0),
        ).alias("margin_residual_bps"),
    )
    if eligible.is_empty():
        raise RuntimeError("no calibration rows have fresh executable quotes")
    records: list[pl.DataFrame] = []
    cells: dict[str, dict[str, ConformalCell]] = {}
    levels = [("global", None)]
    if candidate in (DIRECTION_PRICE, DIRECTION_TIME_PRICE):
        levels.append(("direction_price", "direction_price_group"))
    if candidate == DIRECTION_TIME_PRICE:
        levels.append(("direction_time_price", "direction_time_price_group"))
    for level, column in levels:
        working = eligible.with_columns(
            pl.lit("global").alias("conformal_group")
            if column is None
            else pl.col(column).alias("conformal_group")
        )
        market_blocks = working.group_by(["market_id", "conformal_group"]).agg(
            pl.col("window_start").first(),
            pl.col("probability_residual").max(),
            pl.col("margin_residual_bps").max(),
        ).with_columns(pl.lit(level).alias("conformal_level"))
        records.append(market_blocks)
        level_cells: dict[str, ConformalCell] = {}
        for row in market_blocks.partition_by("conformal_group", as_dict=True).values():
            key = str(row["conformal_group"][0])
            support = row["market_id"].n_unique()
            if level != "global" and support < minimum_cell_markets:
                continue
            level_cells[key] = ConformalCell(
                support,
                conformal_quantile(row["probability_residual"].to_numpy(), alpha),
                conformal_quantile(row["margin_residual_bps"].to_numpy(), alpha),
            )
        cells[level] = level_cells
    artifact = ConformalArtifact(
        ARTIFACT_SCHEMA_VERSION,
        candidate,
        alpha,
        minimum_cell_markets,
        calibration_start,
        calibration_end,
        source_identity,
        prediction_ledger_sha256,
        feature_registry_sha256,
        upstream_artifact_sha256,
        cells,
    )
    validate_artifact(artifact)
    return artifact, pl.concat(records, how="vertical")


def apply_conformal_bounds(frame: pl.DataFrame, artifact: ConformalArtifact) -> pl.DataFrame:
    validate_artifact(artifact)
    grouped = add_group_columns(frame)
    selected_probability = grouped["selected_probability"].to_numpy().astype(float)
    if artifact.candidate == CONTROL:
        lower = selected_probability
        probability_q = np.zeros(grouped.height, dtype=float)
        margin_q = np.zeros(grouped.height, dtype=float)
        fallback = np.full(grouped.height, "not_applicable", dtype=object)
        support = np.zeros(grouped.height, dtype=int)
    else:
        probability_q = np.empty(grouped.height, dtype=float)
        margin_q = np.empty(grouped.height, dtype=float)
        fallback = np.empty(grouped.height, dtype=object)
        support = np.empty(grouped.height, dtype=int)
        direction_price = grouped["direction_price_group"].to_list()
        direction_time_price = grouped["direction_time_price_group"].to_list()
        for index in range(grouped.height):
            cell, used = _lookup_cell(
                artifact,
                str(direction_price[index]),
                str(direction_time_price[index]),
            )
            probability_q[index] = cell.probability_residual_quantile
            margin_q[index] = cell.margin_residual_quantile_bps
            fallback[index] = used
            support[index] = cell.support_markets
        lower = np.clip(selected_probability - probability_q, 0.0, 1.0)
    return grouped.with_columns(
        pl.Series("correctness_lower_bound", lower),
        pl.Series("error_risk_upper_bound", 1.0 - lower),
        pl.Series(
            "conformal_margin_lower",
            grouped["margin_p05_bps"].to_numpy().astype(float) - margin_q,
        ),
        pl.Series(
            "conformal_margin_upper",
            grouped["margin_p95_bps"].to_numpy().astype(float) + margin_q,
        ),
        pl.Series("probability_residual_quantile", probability_q),
        pl.Series("margin_residual_quantile_bps", margin_q),
        pl.Series("calibration_support_markets", support),
        pl.Series("conformal_fallback", fallback),
    )


def _lookup_cell(
    artifact: ConformalArtifact,
    direction_price_group: str,
    direction_time_price_group: str,
) -> tuple[ConformalCell, str]:
    if artifact.candidate == DIRECTION_TIME_PRICE:
        cell = artifact.cells.get("direction_time_price", {}).get(direction_time_price_group)
        if cell is not None:
            return cell, "direction_time_price"
    if artifact.candidate in (DIRECTION_PRICE, DIRECTION_TIME_PRICE):
        cell = artifact.cells.get("direction_price", {}).get(direction_price_group)
        if cell is not None:
            return cell, "direction_price"
    return artifact.cells["global"]["global"], "global"


def apply_admission_contract(
    bounded: pl.DataFrame,
    *,
    reserve_per_share: float,
    stress_slippage_per_share: float,
    minimum_correctness: float,
    maximum_error_risk: float,
    maximum_recovery_ratio: float,
    evaluation_quantity: int,
) -> pl.DataFrame:
    predicted_up = bounded["predicted_up"].to_numpy().astype(bool)
    selected_cost = bounded["selected_cost_5"].to_numpy().astype(float)
    fee_rate = bounded["fee_rate"].fill_null(float("nan")).to_numpy().astype(float)
    fee = fee_rate * selected_cost * (1.0 - selected_cost)
    all_in = selected_cost + fee + reserve_per_share + stress_slippage_per_share
    profit = 1.0 - all_in
    loss = all_in
    recovery = loss / np.maximum(profit, 1e-12)
    lower = bounded["correctness_lower_bound"].to_numpy().astype(float)
    risk = bounded["error_risk_upper_bound"].to_numpy().astype(float)
    payoff_lower = lower * profit - risk * loss
    margin_lower = bounded["conformal_margin_lower"].to_numpy().astype(float)
    margin_upper = bounded["conformal_margin_upper"].to_numpy().astype(float)
    margin_supports = np.where(predicted_up, margin_lower > 0.0, margin_upper < 0.0)
    book_valid = bounded["book_valid"].to_numpy().astype(bool)
    quote_valid = np.isfinite(selected_cost) & np.isfinite(fee) & (selected_cost > 0.0)
    reasons = np.full(bounded.height, "admitted", dtype=object)
    gates = (
        ("stale_or_unavailable_orderbook", book_valid),
        ("unavailable_executable_vwap5", quote_valid),
        ("quote_recovery_geometry", (profit > 0.0) & (recovery <= maximum_recovery_ratio)),
        ("correctness_lower_bound", lower >= minimum_correctness),
        ("error_risk_upper_bound", risk <= maximum_error_risk),
        ("conformal_margin_interval", margin_supports),
        ("stressed_payoff_lower_bound", payoff_lower > 0.0),
    )
    eligible = np.ones(bounded.height, dtype=bool)
    for reason, passed in gates:
        failed = eligible & ~passed
        reasons[failed] = reason
        eligible &= passed
    direction_correct = predicted_up == bounded["label_up"].to_numpy().astype(bool)
    quantity = float(evaluation_quantity)
    gross = quantity * (direction_correct.astype(float) - selected_cost)
    fee_adjusted = gross - quantity * fee
    net = fee_adjusted - quantity * reserve_per_share
    stressed = net - quantity * stress_slippage_per_share
    return bounded.with_columns(
        pl.Series("selected_fee_5", fee),
        pl.Series("stressed_profit_if_correct_per_share", profit),
        pl.Series("stressed_loss_if_wrong_per_share", loss),
        pl.Series("quoted_loss_recovery_ratio", recovery),
        pl.Series("stressed_payoff_lower_bound", payoff_lower),
        pl.Series("margin_interval_supports_side", margin_supports),
        pl.Series("direction_correct", direction_correct),
        pl.Series("abstention_reason", reasons),
        pl.Series("entry_eligible", eligible),
        pl.Series("gross_pnl", gross),
        pl.Series("fee_adjusted_pnl", fee_adjusted),
        pl.Series("net_pnl", net),
        pl.Series("stressed_pnl", stressed),
    )


def earliest_admitted_trades(decisions: pl.DataFrame) -> pl.DataFrame:
    selected = (
        decisions.filter(pl.col("entry_eligible"))
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .unique(subset=["market_id"], keep="first", maintain_order=True)
        .sort(["window_start", "market_id"])
    )
    if selected.height and selected["market_id"].n_unique() != selected.height:
        raise RuntimeError("more than one trade selected per market")
    return selected
