from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .core_config import CORE_ORACLE_SOURCE_CONTRACT, CoreTrainingConfig
from .core_execution import (
    EXECUTION_CONTEXT_SECONDS,
    EXECUTION_DECISION_SECONDS,
    ExecutionEvidenceConfig,
    extract_execution_evidence,
    load_execution_evidence_manifest,
)
from .core_extract import (
    extract_core_source,
    file_sha256,
    load_core_manifest,
    scope_range,
    write_json_atomic,
)
from .core_features import (
    build_core_features,
    feature_destination,
    validate_core_feature_cache,
)

TRAINING_READINESS_SCHEMA_VERSION = (
    "btc-core-oracle-book-training-readiness-v1"
)


def prepare_training_readiness(
    config: CoreTrainingConfig,
    *,
    execution_output_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    _validate_readiness_config(config)
    range_start, range_end = scope_range(config, "pre_holdout")
    extract_core_source(config, "pre_holdout", force=force)
    build_core_features(config, "pre_holdout", force=force)
    execution_config = ExecutionEvidenceConfig(
        range_start=range_start,
        range_end=range_end,
        output_dir=execution_output_dir,
        min_seconds_after_open=EXECUTION_CONTEXT_SECONDS[0],
        max_seconds_after_open=EXECUTION_CONTEXT_SECONDS[-1],
    )
    extract_execution_evidence(execution_config, force=force)
    return generate_training_readiness(
        config,
        execution_config=execution_config,
        output_dir=output_dir,
    )


def generate_training_readiness(
    config: CoreTrainingConfig,
    *,
    execution_config: ExecutionEvidenceConfig,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    _validate_readiness_config(config)
    range_start, range_end = scope_range(config, "pre_holdout")
    if (
        execution_config.range_start != range_start
        or execution_config.range_end != range_end
        or execution_config.min_seconds_after_open
        != EXECUTION_CONTEXT_SECONDS[0]
        or execution_config.max_seconds_after_open
        != EXECUTION_CONTEXT_SECONDS[-1]
        or execution_config.sample_interval_seconds != 5
    ):
        raise ValueError(
            "execution evidence must cover the exact pre-holdout range at "
            "90–140 seconds on a five-second cadence"
        )

    source_manifest = load_core_manifest(config, "pre_holdout")
    feature_metadata = validate_core_feature_cache(
        config,
        "pre_holdout",
    )
    execution_manifest = load_execution_evidence_manifest(execution_config)
    feature_path = feature_destination(config, "pre_holdout")
    feature_frame = pl.read_parquet(
        feature_path,
        columns=[
            "market_id",
            "window_start",
            "observed_at",
            "seconds_elapsed",
            "oracle_model_eligible",
        ],
    )
    _validate_feature_rows(feature_frame, range_start, range_end)

    execution_paths = [
        execution_config.output_dir / partition["path"]
        for partition in execution_manifest["partitions"]
    ]
    execution_frame = pl.concat(
        [
            pl.read_parquet(
                path,
                columns=[
                    "market_id",
                    "window_start",
                    "observed_at",
                    "seconds_elapsed",
                    "strict_both_side_eligible",
                    "strict_both_side_eligible_10",
                ],
            )
            for path in execution_paths
        ],
        how="vertical_relaxed",
    )
    _validate_execution_rows(execution_frame, range_start, range_end)

    oracle_sets = _eligible_sets_by_date_and_second(
        feature_frame,
        eligibility_column="oracle_model_eligible",
        seconds=EXECUTION_DECISION_SECONDS,
    )
    book_five_sets = _eligible_sets_by_date_and_second(
        execution_frame,
        eligibility_column="strict_both_side_eligible",
        seconds=EXECUTION_CONTEXT_SECONDS,
    )
    book_ten_sets = _eligible_sets_by_date_and_second(
        execution_frame,
        eligibility_column="strict_both_side_eligible_10",
        seconds=EXECUTION_CONTEXT_SECONDS,
    )
    core_source_by_date = {
        str(partition["path"]).removesuffix(".parquet"): int(
            partition["markets"]
        )
        for partition in source_manifest["partitions"]
    }
    core_complete_by_date = _daily_count_map(
        feature_metadata["daily_core_counts"]
    )
    oracle_metadata = feature_metadata.get("oracle")
    if not isinstance(oracle_metadata, dict):
        raise TypeError(
            "oracle feature metadata is missing its readiness cohort"
        )
    oracle_complete_by_date = _daily_count_map(
        oracle_metadata["daily_complete_counts"]
    )

    daily: list[dict[str, Any]] = []
    current = range_start
    while current < range_end:
        day_end = min(current + timedelta(days=1), range_end)
        date = current.date().isoformat()
        expected_markets = int(
            (day_end - current).total_seconds() // 300
        )
        raw_five = {
            str(second): len(book_five_sets.get((date, second), set()))
            for second in EXECUTION_CONTEXT_SECONDS
        }
        raw_ten = {
            str(second): len(book_ten_sets.get((date, second), set()))
            for second in EXECUTION_CONTEXT_SECONDS
        }
        complete_11 = set.intersection(
            *[
                book_ten_sets.get((date, second), set())
                for second in EXECUTION_CONTEXT_SECONDS
            ]
        )
        point_sets = {
            second: (
                book_ten_sets.get((date, second - 5), set())
                & book_ten_sets.get((date, second), set())
            )
            for second in EXECUTION_DECISION_SECONDS
        }
        common_sets = {
            second: (
                point_sets[second]
                & oracle_sets.get((date, second), set())
            )
            for second in EXECUTION_DECISION_SECONDS
        }
        daily.append(
            {
                "date": date,
                "expected_markets": expected_markets,
                "core_source_markets": core_source_by_date.get(date, 0),
                "core_complete_markets": core_complete_by_date.get(date, 0),
                "core_oracle_complete_markets": (
                    oracle_complete_by_date.get(date, 0)
                ),
                "book_complete_11_point_markets": len(complete_11),
                "book_raw_strict_markets_by_second": raw_ten,
                "book_raw_strict_five_share_markets_by_second": raw_five,
                "book_raw_strict_ten_share_markets_by_second": raw_ten,
                "book_point_qualified_markets_by_second": {
                    str(second): len(point_sets[second])
                    for second in EXECUTION_DECISION_SECONDS
                },
                "common_oracle_book_point_qualified_markets_by_second": {
                    str(second): len(common_sets[second])
                    for second in EXECUTION_DECISION_SECONDS
                },
            }
        )
        current = day_end

    totals = _aggregate_daily_readiness(daily)
    cohorts = _aggregate_readiness_cohorts(config, daily)
    source_manifest_path = (
        config.paths.source_data / "manifest-pre_holdout.json"
    )
    feature_metadata_path = feature_path.with_suffix(".metadata.json")
    execution_manifest_path = execution_config.output_dir / "manifest.json"
    payload: dict[str, Any] = {
        "schema_version": TRAINING_READINESS_SCHEMA_VERSION,
        "range_start": range_start.isoformat(),
        "range_end": range_end.isoformat(),
        "range_semantics": "half_open",
        "source_contract": config.data.source_contract,
        "context_seconds": list(EXECUTION_CONTEXT_SECONDS),
        "decision_seconds": list(EXECUTION_DECISION_SECONDS),
        "book_point_qualification": (
            "strict ten-share eligibility at both t-5 and t"
        ),
        "book_complete_11_point_qualification": (
            "strict ten-share eligibility at every exact context second"
        ),
        "freshness_seconds": execution_config.freshness_seconds,
        "input_hashes": {
            "training_config": file_sha256(config.source_path),
            "core_source_manifest": file_sha256(source_manifest_path),
            "feature_metadata": file_sha256(feature_metadata_path),
            "feature_parquet": file_sha256(feature_path),
            "execution_manifest": file_sha256(
                execution_manifest_path
            ),
        },
        "input_contracts": {
            "core_source_schema_version": source_manifest[
                "source_schema_version"
            ],
            "oracle_source_schema_version": source_manifest[
                "oracle_source_schema_version"
            ],
            "feature_schema_version": feature_metadata[
                "feature_schema_version"
            ],
            "oracle_feature_schema_version": feature_metadata[
                "candidate_feature_schema_versions"
            ]["histogram_mature_reversal_oracle"],
            "execution_source_contract": execution_manifest[
                "source_contract"
            ],
            "execution_source_schema_version": execution_manifest[
                "source_schema_version"
            ],
        },
        "daily": daily,
        "totals": totals,
        "cohorts": cohorts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "training-readiness.json"
    markdown_path = output_dir / "training-readiness.md"
    write_json_atomic(json_path, payload)
    _write_text_atomic(markdown_path, _render_markdown(payload))
    return json_path, markdown_path, payload


def _validate_readiness_config(config: CoreTrainingConfig) -> None:
    if config.data.source_contract != CORE_ORACLE_SOURCE_CONTRACT:
        raise ValueError(
            "training readiness requires the btc_core_oracle_v1 contract"
        )
    if config.data.sample_interval_seconds != 5:
        raise ValueError(
            "training readiness requires a five-second decision cadence"
        )
    if (
        config.data.min_seconds_after_open
        != EXECUTION_DECISION_SECONDS[0]
        or 300 - config.data.min_seconds_before_close
        != EXECUTION_DECISION_SECONDS[-1]
    ):
        raise ValueError(
            "training readiness requires the exact 120–140 second "
            "decision window"
        )
    range_start, range_end = scope_range(config, "pre_holdout")
    for name, value in (
        ("range_start", range_start),
        ("range_end", range_end),
    ):
        if value.utcoffset() != timedelta(0):
            raise ValueError(f"{name} must be UTC")
        if value.astimezone(UTC).time() != datetime.min.time():
            raise ValueError(f"{name} must align to a UTC day")


def _validate_feature_rows(
    frame: pl.DataFrame,
    range_start: datetime,
    range_end: datetime,
) -> None:
    _validate_row_keys_and_times(
        frame,
        range_start,
        range_end,
        allowed_seconds=set(EXECUTION_DECISION_SECONDS),
        source_name="oracle feature",
    )
    if frame.filter(~pl.col("oracle_model_eligible")).height:
        raise RuntimeError(
            "oracle feature cache contains ineligible decision rows"
        )
    expected_seconds = set(EXECUTION_DECISION_SECONDS)
    for row in (
        frame.group_by("market_id")
        .agg(pl.col("seconds_elapsed").unique())
        .to_dicts()
    ):
        if set(row["seconds_elapsed"]) != expected_seconds:
            raise RuntimeError(
                "oracle feature cache contains an incomplete decision market"
            )


def _validate_execution_rows(
    frame: pl.DataFrame,
    range_start: datetime,
    range_end: datetime,
) -> None:
    _validate_row_keys_and_times(
        frame,
        range_start,
        range_end,
        allowed_seconds=set(EXECUTION_CONTEXT_SECONDS),
        source_name="execution evidence",
    )
    if frame.filter(
        pl.col("strict_both_side_eligible_10")
        & ~pl.col("strict_both_side_eligible")
    ).height:
        raise RuntimeError(
            "execution evidence has ten-share eligibility without "
            "five-share eligibility"
        )


def _validate_row_keys_and_times(
    frame: pl.DataFrame,
    range_start: datetime,
    range_end: datetime,
    *,
    allowed_seconds: set[int],
    source_name: str,
) -> None:
    duplicate_keys = (
        frame.group_by(["market_id", "observed_at"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate_keys.height:
        raise RuntimeError(
            f"{source_name} contains duplicate market/timestamp keys"
        )
    invalid = frame.filter(
        (pl.col("window_start") < range_start)
        | (pl.col("window_start") >= range_end)
        | ~pl.col("seconds_elapsed").is_in(sorted(allowed_seconds))
        | (
            pl.col("observed_at")
            != (
                pl.col("window_start")
                + pl.duration(seconds=pl.col("seconds_elapsed"))
            )
        )
    )
    if invalid.height:
        raise RuntimeError(
            f"{source_name} contains rows outside its exact time contract"
        )


def _eligible_sets_by_date_and_second(
    frame: pl.DataFrame,
    *,
    eligibility_column: str,
    seconds: tuple[int, ...],
) -> dict[tuple[str, int], set[str]]:
    result: dict[tuple[str, int], set[str]] = {}
    eligible = frame.filter(pl.col(eligibility_column))
    for row in eligible.select(
        "market_id",
        "window_start",
        "seconds_elapsed",
    ).iter_rows(named=True):
        second = int(row["seconds_elapsed"])
        if second not in seconds:
            continue
        date = row["window_start"].date().isoformat()
        result.setdefault((date, second), set()).add(
            str(row["market_id"])
        )
    return result


def _daily_count_map(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {str(row["date"]): int(row["markets"]) for row in rows}


def _aggregate_daily_readiness(
    daily: list[dict[str, Any]],
) -> dict[str, Any]:
    scalar_keys = (
        "expected_markets",
        "core_source_markets",
        "core_complete_markets",
        "core_oracle_complete_markets",
        "book_complete_11_point_markets",
    )
    totals: dict[str, Any] = {
        key: sum(int(row[key]) for row in daily)
        for key in scalar_keys
    }
    for key, seconds in (
        (
            "book_raw_strict_markets_by_second",
            EXECUTION_CONTEXT_SECONDS,
        ),
        (
            "book_raw_strict_five_share_markets_by_second",
            EXECUTION_CONTEXT_SECONDS,
        ),
        (
            "book_raw_strict_ten_share_markets_by_second",
            EXECUTION_CONTEXT_SECONDS,
        ),
        (
            "book_point_qualified_markets_by_second",
            EXECUTION_DECISION_SECONDS,
        ),
        (
            "common_oracle_book_point_qualified_markets_by_second",
            EXECUTION_DECISION_SECONDS,
        ),
    ):
        totals[key] = {
            str(second): sum(
                int(row[key][str(second)]) for row in daily
            )
            for second in seconds
        }
    expected = totals["expected_markets"]
    totals["coverage_rates"] = {
        "core_complete": (
            totals["core_complete_markets"] / expected
            if expected
            else 0.0
        ),
        "core_oracle_complete": (
            totals["core_oracle_complete_markets"] / expected
            if expected
            else 0.0
        ),
        "book_complete_11_point": (
            totals["book_complete_11_point_markets"] / expected
            if expected
            else 0.0
        ),
        "common_point_qualified_by_second": {
            str(second): (
                totals[
                    "common_oracle_book_point_qualified_markets_by_second"
                ][str(second)]
                / expected
                if expected
                else 0.0
            )
            for second in EXECUTION_DECISION_SECONDS
        },
    }
    return totals


def _aggregate_readiness_cohorts(
    config: CoreTrainingConfig,
    daily: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    boundaries = {
        "development": (
            config.split.development_start,
            config.split.development_end,
        ),
        "probability_calibration": (
            config.split.probability_calibration_start,
            config.split.probability_calibration_end,
        ),
        "policy_selection": (
            config.split.policy_selection_start,
            config.split.policy_selection_end,
        ),
        "holdout": (
            config.split.holdout_start,
            config.split.holdout_end,
        ),
    }
    cohorts: dict[str, dict[str, Any]] = {}
    for name, (start, end) in boundaries.items():
        rows = [
            row
            for row in daily
            if start.date().isoformat() <= row["date"] < end.date().isoformat()
        ]
        cohorts[name] = {
            "range_start": start.isoformat(),
            "range_end": end.isoformat(),
            "totals": _aggregate_daily_readiness(rows),
        }
    return cohorts


def _render_markdown(payload: dict[str, Any]) -> str:
    totals = payload["totals"]
    lines = [
        "# BTC core + oracle + book training readiness",
        "",
        (
            f"Range: `{payload['range_start']}` through "
            f"`{payload['range_end']}` (half-open)."
        ),
        "",
        (
            "Book-complete markets require strict ten-share executable "
            "evidence at all 11 exact points from 90 through 140 seconds."
        ),
        "",
        (
            "Point-qualified markets require strict ten-share evidence at "
            "both `t-5` and `t`; common counts additionally require the "
            "causal oracle feature row at `t`."
        ),
        "",
        "## Totals",
        "",
        "| Dimension | Markets | Coverage |",
        "|---|---:|---:|",
    ]
    for label, key, rate_key in (
        ("Expected", "expected_markets", None),
        ("Core complete", "core_complete_markets", "core_complete"),
        (
            "Core + oracle complete",
            "core_oracle_complete_markets",
            "core_oracle_complete",
        ),
        (
            "Book complete (11 points)",
            "book_complete_11_point_markets",
            "book_complete_11_point",
        ),
    ):
        rate = (
            1.0
            if rate_key is None
            else totals["coverage_rates"][rate_key]
        )
        lines.append(
            f"| {label} | {totals[key]:,} | {rate:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Common oracle/book point-qualified coverage",
            "",
            "| Decision second | Markets | Coverage |",
            "|---:|---:|---:|",
        ]
    )
    for second in EXECUTION_DECISION_SECONDS:
        count = totals[
            "common_oracle_book_point_qualified_markets_by_second"
        ][str(second)]
        rate = totals["coverage_rates"][
            "common_point_qualified_by_second"
        ][str(second)]
        lines.append(f"| {second} | {count:,} | {rate:.2%} |")
    lines.extend(
        [
            "",
            "## Chronological cohorts",
            "",
            (
                "| Cohort | Range | Core + oracle | Book 11 | Common 120 | "
                "Common 125 | Common 130 | Common 135 | Common 140 |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in (
        "development",
        "probability_calibration",
        "policy_selection",
        "holdout",
    ):
        cohort = payload["cohorts"][name]
        cohort_totals = cohort["totals"]
        common = cohort_totals[
            "common_oracle_book_point_qualified_markets_by_second"
        ]
        lines.append(
            "| {name} | `{start}`–`{end}` | {oracle:,} | {book:,} | "
            "{c120:,} | {c125:,} | {c130:,} | {c135:,} | {c140:,} |".format(
                name=name.replace("_", " ").title(),
                start=cohort["range_start"][:10],
                end=cohort["range_end"][:10],
                oracle=cohort_totals["core_oracle_complete_markets"],
                book=cohort_totals["book_complete_11_point_markets"],
                c120=common["120"],
                c125=common["125"],
                c130=common["130"],
                c135=common["135"],
                c140=common["140"],
            )
        )
    lines.extend(
        [
            "",
            "## Daily readiness",
            "",
            (
                "| Date | Expected | Core | Core + oracle | Book 11 | "
                "Common 120 | Common 125 | Common 130 | Common 135 | "
                "Common 140 |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["daily"]:
        common = row[
            "common_oracle_book_point_qualified_markets_by_second"
        ]
        lines.append(
            "| {date} | {expected:,} | {core:,} | {oracle:,} | "
            "{book:,} | {c120:,} | {c125:,} | {c130:,} | {c135:,} | "
            "{c140:,} |".format(
                date=row["date"],
                expected=row["expected_markets"],
                core=row["core_complete_markets"],
                oracle=row["core_oracle_complete_markets"],
                book=row["book_complete_11_point_markets"],
                c120=common["120"],
                c125=common["125"],
                c130=common["130"],
                c135=common["135"],
                c140=common["140"],
            )
        )
    zero_book_dates = [
        row["date"]
        for row in payload["daily"]
        if row["book_complete_11_point_markets"] == 0
    ]
    lines.extend(
        [
            "",
            "## Zero-book dates",
            "",
            (
                ", ".join(f"`{date}`" for date in zero_book_dates)
                if zero_book_dates
                else "None."
            ),
            "",
            (
                "The JSON companion contains exact five-share and "
                "ten-share raw strict counts for every context second and "
                "all input hashes."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text)
    temporary.replace(path)
