from __future__ import annotations

import hashlib
import html
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from .champion_vwap_config import ChampionVwapBenchmarkConfig
from .core_extract import file_sha256, write_json_atomic
from .runtime_export import score_runtime_model

CONTROL = "frozen_champion"
VWAP5 = "vwap5_conditioned_calibration"
VWAP5_DEPTH = "vwap5_vwap10_depth_conditioned_calibration"
CHALLENGERS = (VWAP5, VWAP5_DEPTH)
CANDIDATES = (CONTROL, *CHALLENGERS)
SCHEMA_VERSION = "btc-champion-vwap-calibration-benchmark-v1"
PRICE_BANDS = (
    ("under_0_70", 0.00, 0.70),
    ("0_70_to_0_80", 0.70, 0.80),
    ("0_80_to_0_85", 0.80, 0.85),
    ("0_85_to_0_90", 0.85, 0.90),
    ("0_90_to_0_92", 0.90, 0.92),
    ("0_92_and_over", 0.92, 1.01),
)
DEPTH_BANDS = (
    ("under_0_005", -1.0, 0.005),
    ("0_005_to_0_01", 0.005, 0.01),
    ("0_01_to_0_02", 0.01, 0.02),
    ("0_02_and_over", 0.02, 1.01),
)


@dataclass(frozen=True)
class ConditionedCalibrator:
    candidate: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    iterations: int
    converged: bool
    fit_rows: int
    fit_start: str
    fit_end: str

    def probability(self, frame: pl.DataFrame) -> np.ndarray:
        matrix = _finite_matrix(frame, self.feature_names)
        standardized = (matrix - np.asarray(self.means)) / np.asarray(self.scales)
        logit = standardized @ np.asarray(self.coefficients) + self.intercept
        return _sigmoid(logit)


def run_champion_vwap_benchmark(
    config: ChampionVwapBenchmarkConfig,
) -> tuple[Path, dict[str, Any]]:
    print("champion-vwap: validating frozen artifact and causal caches", flush=True)
    decisions, lineage = load_qualified_champion_decisions(config)
    calibration = _range(
        decisions,
        config.data.out_of_sample_score_start,
        config.data.calibration_end,
    )
    holdout = _range(decisions, config.data.calibration_end, config.data.holdout_end)
    if calibration.height < config.promotion.minimum_calibration_rows:
        raise RuntimeError("qualified OOS calibration cohort is below the row gate")
    if holdout.height < config.promotion.minimum_holdout_rows:
        raise RuntimeError("qualified final holdout cohort is below the row gate")

    print("champion-vwap: evaluating expanding chronological folds", flush=True)
    fold_results: dict[str, Any] = {}
    for fold in config.folds:
        fit = _range(
            decisions,
            config.data.out_of_sample_score_start,
            fold.fit_end,
        )
        evaluation = _range(decisions, fold.fit_end, fold.evaluation_end)
        models = fit_challenger_calibrators(fit, config)
        scored = score_candidates(evaluation, models, config.model.confidence_threshold)
        fold_results[fold.name] = {
            "fit": _cohort(fit),
            "evaluation": _cohort(evaluation),
            "calibrators": {name: asdict(model) for name, model in models.items()},
            "candidates": {
                name: candidate_metrics(frame, eligible_markets=evaluation.height)
                for name, frame in scored.items()
            },
        }

    print("champion-vwap: fitting final calibrators before holdout access", flush=True)
    final_models = fit_challenger_calibrators(calibration, config)
    scored_calibration = score_candidates(
        calibration,
        final_models,
        config.model.confidence_threshold,
    )
    scored_holdout = score_candidates(
        holdout,
        final_models,
        config.model.confidence_threshold,
    )
    calibration_metrics = {
        name: candidate_metrics(frame, eligible_markets=calibration.height)
        for name, frame in scored_calibration.items()
    }
    holdout_metrics = {
        name: candidate_metrics(frame, eligible_markets=holdout.height)
        for name, frame in scored_holdout.items()
    }
    promotion = promotion_decision(holdout_metrics, fold_results, config)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_records: dict[str, Any] = {}
    for candidate, model in final_models.items():
        model_key = _calibrator_model_key(config.champion_model_key, candidate)
        payload = {
            "schema_version": "capitonic-btc-conditioned-calibrator-v1",
            "model_key": model_key,
            "base_model_key": config.champion_model_key,
            "base_model_sha256": config.champion_model_sha256,
            "source_process_id": str(config.source_process_id),
            "direction_locked_to_base_model": True,
            "calibrator": asdict(model),
            "confidence_threshold": config.model.confidence_threshold,
            "promotion_qualified": promotion["candidates"][candidate]["passed"],
            "runtime_exported": False,
        }
        artifact_path = run_dir / f"{candidate}-calibrator.json"
        write_json_atomic(artifact_path, payload)
        artifact_records[candidate] = {
            "model_key": model_key,
            "path": str(artifact_path),
            "sha256": file_sha256(artifact_path),
        }
    for candidate, frame in scored_holdout.items():
        frame.write_parquet(run_dir / f"{candidate}-holdout.parquet", compression="zstd")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "challenger_selected" if promotion["selected_candidate"] else "champion_retained",
        "evaluation_note": config.evaluation_note,
        "source_process_id": str(config.source_process_id),
        "champion": {
            "model_key": config.champion_model_key,
            "model_sha256": config.champion_model_sha256,
            "feature_schema_sha256": config.champion_feature_schema_sha256,
            "directional_classifier_frozen": True,
            "global_platt_calibration_frozen_for_control": True,
            "confidence_threshold": config.model.confidence_threshold,
            "first_crossing_policy": {
                "minimum_seconds_after_open": config.model.minimum_seconds_after_open,
                "maximum_seconds_after_open": config.model.maximum_seconds_after_open,
                "cadence_seconds": config.model.cadence_seconds,
            },
        },
        "data_contract": {
            "source_range_start": config.data.source_range_start.isoformat(),
            "source_range_end": config.data.source_range_end.isoformat(),
            "order_book_coverage_start": config.data.order_book_coverage_start.isoformat(),
            "out_of_sample_score_start": config.data.out_of_sample_score_start.isoformat(),
            "calibration_end": config.data.calibration_end.isoformat(),
            "holdout_end": config.data.holdout_end.isoformat(),
            "one_row_per_champion_first_crossing": True,
            "identical_rows_for_all_candidates": True,
            "vwap_imputation": False,
            "missingness_features": False,
            "direction_override_allowed": False,
            "quantity": config.model.quantity,
            "fee_formula": "fee_rate * price * (1 - price)",
            "ten_share_economics_separate": True,
        },
        "lineage": lineage,
        "cohorts": {
            "all_qualified_oos_decisions": _cohort(decisions),
            "calibration": _cohort(calibration),
            "holdout": _cohort(holdout),
        },
        "calibrator_artifacts": artifact_records,
        "calibration_diagnostics": calibration_metrics,
        "chronological_folds": fold_results,
        "holdout": holdout_metrics,
        "promotion": promotion,
    }
    write_json_atomic(run_dir / "benchmark.json", result)
    (run_dir / "report.html").write_text(_report_html(result))
    print(
        "champion-vwap: complete; selected="
        f"{promotion['selected_candidate'] or 'none'}",
        flush=True,
    )
    return run_dir, result


def load_qualified_champion_decisions(
    config: ChampionVwapBenchmarkConfig,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    model = json.loads(config.paths.champion_model.read_text())
    manifest = json.loads(config.paths.champion_manifest.read_text())
    _validate_champion_payload(model, manifest, config)
    metadata = json.loads(config.paths.feature_metadata.read_text())
    if metadata["feature_file_sha256"] != file_sha256(config.paths.features):
        raise RuntimeError("feature cache hash mismatch")
    build_contract = metadata["build_contract"]
    if (
        _utc(build_contract["range_start"]) != config.data.source_range_start
        or _utc(build_contract["range_end"]) != config.data.source_range_end
    ):
        raise RuntimeError("feature cache range does not match the benchmark contract")

    execution_manifest_path = config.paths.execution_evidence / "manifest.json"
    execution_manifest = json.loads(execution_manifest_path.read_text())
    if (
        _utc(execution_manifest["range_start"]) != config.data.source_range_start
        or _utc(execution_manifest["range_end"]) != config.data.source_range_end
    ):
        raise RuntimeError("execution cache range does not match the benchmark contract")
    partitions = []
    for record in execution_manifest["partitions"]:
        path = config.paths.execution_evidence / record["path"]
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"execution partition hash mismatch: {path.name}")
        partitions.append(path)

    execution = (
        pl.scan_parquet(partitions)
        .filter(
            pl.col("strict_both_side_eligible_10")
            & (pl.col("window_start") >= config.data.out_of_sample_score_start)
            & (pl.col("window_start") < config.data.holdout_end)
        )
        .select(
            "market_id",
            "window_start",
            "observed_at",
            "seconds_elapsed",
            "fee_rate",
            "up_ask_vwap_5",
            "up_ask_vwap_10",
            "down_ask_vwap_5",
            "down_ask_vwap_10",
        )
        .collect()
        .unique(subset=["market_id", "observed_at"])
    )
    if execution.is_empty():
        raise RuntimeError("strict VWAP5/VWAP10 evidence cohort is empty")
    market_ids = execution["market_id"].unique()
    feature_names = tuple(model["features"]["names"])
    features = (
        pl.scan_parquet(config.paths.features)
        .filter(
            (pl.col("window_start") >= config.data.out_of_sample_score_start)
            & (pl.col("window_start") < config.data.holdout_end)
            & pl.col("market_id").is_in(market_ids.implode())
        )
        .select(
            "market_id",
            "window_start",
            "observed_at",
            "seconds_elapsed",
            "label_up",
            *feature_names,
        )
        .collect()
        .sort(["market_id", "seconds_elapsed", "observed_at"])
    )
    first_crossings = _first_champion_crossings(features, model, feature_names)
    joined = first_crossings.join(
        execution,
        on=["market_id", "window_start", "observed_at", "seconds_elapsed"],
        how="inner",
        validate="1:1",
    )
    if joined["market_id"].n_unique() != joined.height:
        raise RuntimeError("qualified champion decisions must contain one row per market")
    qualified = _attach_execution_economics(joined, config.model.quantity)
    return qualified.sort(["window_start", "market_id"]), {
        "configuration_sha256": file_sha256(config.source_path),
        "feature_file": str(config.paths.features),
        "feature_file_sha256": metadata["feature_file_sha256"],
        "feature_metadata_sha256": file_sha256(config.paths.feature_metadata),
        "execution_manifest": str(execution_manifest_path),
        "execution_manifest_sha256": file_sha256(execution_manifest_path),
        "execution_partitions_verified": len(partitions),
        "champion_model_file": str(config.paths.champion_model),
        "champion_manifest_file": str(config.paths.champion_manifest),
        "champion_manifest_sha256": file_sha256(config.paths.champion_manifest),
        "markets_with_any_strict_execution_evidence": execution["market_id"].n_unique(),
        "champion_first_crossings": first_crossings.height,
        "book_qualified_champion_first_crossings": qualified.height,
    }


def fit_challenger_calibrators(
    frame: pl.DataFrame,
    config: ChampionVwapBenchmarkConfig,
) -> dict[str, ConditionedCalibrator]:
    if frame.height < config.promotion.minimum_calibration_rows:
        raise RuntimeError("challenger fit cohort is below the calibration row gate")
    return {
        VWAP5: fit_conditioned_calibrator(
            frame,
            candidate=VWAP5,
            feature_names=("champion_selected_raw_logit", "selected_ask_vwap_5"),
            c=config.model.logistic_c,
            maximum_iterations=config.model.maximum_iterations,
        ),
        VWAP5_DEPTH: fit_conditioned_calibrator(
            frame,
            candidate=VWAP5_DEPTH,
            feature_names=(
                "champion_selected_raw_logit",
                "selected_ask_vwap_5",
                "vwap10_minus_vwap5",
            ),
            c=config.model.logistic_c,
            maximum_iterations=config.model.maximum_iterations,
        ),
    }


def fit_conditioned_calibrator(
    frame: pl.DataFrame,
    *,
    candidate: str,
    feature_names: tuple[str, ...],
    c: float,
    maximum_iterations: int,
) -> ConditionedCalibrator:
    matrix = _finite_matrix(frame, feature_names)
    labels = frame["correct"].cast(pl.Int8).to_numpy()
    if len(np.unique(labels)) != 2:
        raise RuntimeError("calibrator fitting requires both win and loss rows")
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    standardized = (matrix - means) / scales
    estimator = LogisticRegression(
        C=c,
        solver="lbfgs",
        max_iter=maximum_iterations,
        tol=1e-10,
        random_state=20260801,
    )
    estimator.fit(standardized, labels)
    iterations = int(estimator.n_iter_[0])
    return ConditionedCalibrator(
        candidate=candidate,
        feature_names=feature_names,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in estimator.coef_[0]),
        intercept=float(estimator.intercept_[0]),
        iterations=iterations,
        converged=iterations < maximum_iterations,
        fit_rows=frame.height,
        fit_start=frame["window_start"].min().isoformat(),
        fit_end=frame["window_start"].max().isoformat(),
    )


def score_candidates(
    frame: pl.DataFrame,
    models: dict[str, ConditionedCalibrator],
    confidence_threshold: float,
) -> dict[str, pl.DataFrame]:
    output = {
        CONTROL: frame.with_columns(
            pl.lit(CONTROL).alias("candidate"),
            pl.col("champion_selected_probability").alias(
                "calibrated_probability_correct"
            ),
            pl.lit(True).alias("policy_selected"),
        )
    }
    for candidate, model in models.items():
        probability = model.probability(frame)
        output[candidate] = frame.with_columns(
            pl.lit(candidate).alias("candidate"),
            pl.Series("calibrated_probability_correct", probability),
            pl.Series("policy_selected", probability >= confidence_threshold),
        )
    return output


def candidate_metrics(frame: pl.DataFrame, *, eligible_markets: int) -> dict[str, Any]:
    selected = frame.filter(pl.col("policy_selected"))
    probability = frame["calibrated_probability_correct"].to_numpy()
    labels = frame["correct"].cast(pl.Int8).to_numpy()
    five_share = _execution_metrics(selected, "realized_net_pnl_5")
    ten_share = _execution_metrics(selected, "realized_net_pnl_10")
    up = selected.filter(pl.col("label_up") == 1)
    down = selected.filter(pl.col("label_up") == 0)
    return {
        "eligible_markets": eligible_markets,
        "trades": selected.height,
        "trade_coverage": selected.height / eligible_markets if eligible_markets else 0.0,
        "accuracy": float(selected["correct"].mean()) if selected.height else None,
        "up_recall": float(up["correct"].mean()) if up.height else None,
        "down_recall": float(down["correct"].mean()) if down.height else None,
        "expected_calibration_error": _ece(probability, labels),
        "five_share": {
            **five_share,
            "net_pnl_per_eligible_market": (
                five_share["net_pnl"] / eligible_markets if eligible_markets else 0.0
            ),
        },
        "ten_share_reporting_only": {
            **ten_share,
            "net_pnl_per_eligible_market": (
                ten_share["net_pnl"] / eligible_markets if eligible_markets else 0.0
            ),
        },
        "price": _price_diagnostics(selected),
        "depth": _depth_diagnostics(selected),
    }


def promotion_decision(
    holdout: dict[str, dict[str, Any]],
    folds: dict[str, Any],
    config: ChampionVwapBenchmarkConfig,
) -> dict[str, Any]:
    control = holdout[CONTROL]
    candidates: dict[str, Any] = {}
    for candidate in CHALLENGERS:
        result = holdout[candidate]
        improving_folds = sum(
            _fold_improves(
                fold["candidates"][candidate],
                fold["candidates"][CONTROL],
                config,
            )
            for fold in folds.values()
        )
        checks = [
            _check(
                "five-share net expectancy per eligible market improves",
                result["five_share"]["net_pnl_per_eligible_market"],
                control["five_share"]["net_pnl_per_eligible_market"],
                ">",
            ),
            _check(
                "profit factor improves",
                _comparable_profit_factor(result["five_share"]),
                _comparable_profit_factor(control["five_share"]),
                ">",
            ),
            _check(
                "average-loss recovery requirement improves",
                _comparable_recovery(result["five_share"]),
                _comparable_recovery(control["five_share"]),
                "<",
            ),
            {
                "name": "maximum drawdown or worst-one-percent loss improves",
                "observed": {
                    "maximum_drawdown": result["five_share"]["maximum_drawdown"],
                    "worst_one_percent_mean": result["five_share"][
                        "worst_one_percent_mean"
                    ],
                },
                "required": {
                    "maximum_drawdown_below": control["five_share"]["maximum_drawdown"],
                    "worst_one_percent_mean_above": control["five_share"][
                        "worst_one_percent_mean"
                    ],
                },
                "passed": (
                    result["five_share"]["maximum_drawdown"]
                    < control["five_share"]["maximum_drawdown"]
                    or result["five_share"]["worst_one_percent_mean"]
                    > control["five_share"]["worst_one_percent_mean"]
                ),
            },
            _check(
                "accuracy remains within tolerance",
                result["accuracy"] if result["accuracy"] is not None else -1.0,
                (control["accuracy"] or 0.0) - config.promotion.accuracy_tolerance,
                ">=",
            ),
            _check(
                "champion qualified trade coverage retained",
                result["trade_coverage"],
                config.promotion.minimum_champion_coverage,
                ">=",
            ),
            _check(
                "multiple chronological periods improve",
                improving_folds,
                config.promotion.minimum_improving_folds,
                ">=",
            ),
        ]
        candidates[candidate] = {
            "improving_folds": improving_folds,
            "checks": checks,
            "passed": all(check["passed"] for check in checks),
        }
    qualified = [name for name in CHALLENGERS if candidates[name]["passed"]]
    selected = (
        max(
            qualified,
            key=lambda name: holdout[name]["five_share"]["net_pnl_per_eligible_market"],
        )
        if qualified
        else None
    )
    return {
        "selected_candidate": selected,
        "champion_retained": selected is None,
        "runtime_changed": False,
        "trading_process_changed": False,
        "candidates": candidates,
    }


def _first_champion_crossings(
    frame: pl.DataFrame,
    model: dict[str, Any],
    feature_names: tuple[str, ...],
) -> pl.DataFrame:
    records: list[dict[str, Any]] = []
    for market in frame.partition_by("market_id", maintain_order=True):
        for row in market.iter_rows(named=True):
            prediction = score_runtime_model(
                model,
                [row[name] for name in feature_names],
                seconds_elapsed=int(row["seconds_elapsed"]),
            )
            if prediction["action"] == "no_trade":
                continue
            selected_up = prediction["action"] == "up"
            raw_logit = float(prediction["raw_logit"])
            probability_up = float(prediction["probability_up"])
            records.append(
                {
                    "market_id": row["market_id"],
                    "window_start": row["window_start"],
                    "observed_at": row["observed_at"],
                    "seconds_elapsed": row["seconds_elapsed"],
                    "label_up": row["label_up"],
                    "champion_selected_up": selected_up,
                    "champion_raw_logit": raw_logit,
                    "champion_selected_raw_logit": raw_logit if selected_up else -raw_logit,
                    "champion_probability_up": probability_up,
                    "champion_selected_probability": (
                        probability_up if selected_up else 1.0 - probability_up
                    ),
                }
            )
            break
    if not records:
        raise RuntimeError("champion produced no first-confidence crossings")
    return pl.DataFrame(records)


def _attach_execution_economics(frame: pl.DataFrame, quantity: float) -> pl.DataFrame:
    attached = frame.with_columns(
        pl.when(pl.col("champion_selected_up"))
        .then(pl.col("up_ask_vwap_5"))
        .otherwise(pl.col("down_ask_vwap_5"))
        .alias("selected_ask_vwap_5"),
        pl.when(pl.col("champion_selected_up"))
        .then(pl.col("up_ask_vwap_10"))
        .otherwise(pl.col("down_ask_vwap_10"))
        .alias("selected_ask_vwap_10"),
        (pl.col("champion_selected_up") == pl.col("label_up").cast(pl.Boolean)).alias(
            "correct"
        ),
    ).with_columns(
        (pl.col("selected_ask_vwap_10") - pl.col("selected_ask_vwap_5")).alias(
            "vwap10_minus_vwap5"
        ),
        (
            pl.col("fee_rate")
            * pl.col("selected_ask_vwap_5")
            * (1.0 - pl.col("selected_ask_vwap_5"))
        ).alias("fee_per_share_5"),
        (
            pl.col("fee_rate")
            * pl.col("selected_ask_vwap_10")
            * (1.0 - pl.col("selected_ask_vwap_10"))
        ).alias("fee_per_share_10"),
    ).with_columns(
        (
            (
                pl.col("correct").cast(pl.Float64)
                - pl.col("selected_ask_vwap_5")
                - pl.col("fee_per_share_5")
            )
            * quantity
        ).alias("realized_net_pnl_5"),
        (
            (
                pl.col("correct").cast(pl.Float64)
                - pl.col("selected_ask_vwap_10")
                - pl.col("fee_per_share_10")
            )
            * 10.0
        ).alias("realized_net_pnl_10"),
    )
    required = (
        "selected_ask_vwap_5",
        "selected_ask_vwap_10",
        "vwap10_minus_vwap5",
        "fee_rate",
    )
    _finite_matrix(attached, required)
    invalid = attached.filter(
        (pl.col("selected_ask_vwap_5") <= 0)
        | (pl.col("selected_ask_vwap_5") > 1)
        | (pl.col("selected_ask_vwap_10") <= 0)
        | (pl.col("selected_ask_vwap_10") > 1)
        | (pl.col("vwap10_minus_vwap5") < -1e-12)
    )
    if invalid.height:
        raise RuntimeError("qualified VWAP rows contain invalid executable prices")
    return attached


def _execution_metrics(selected: pl.DataFrame, pnl_column: str) -> dict[str, Any]:
    pnl = selected[pnl_column].to_numpy() if selected.height else np.asarray([], dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    average_win = float(wins.mean()) if len(wins) else None
    average_loss = float(losses.mean()) if len(losses) else None
    cumulative = np.cumsum(pnl)
    drawdown = np.maximum.accumulate(np.r_[0.0, cumulative]) - np.r_[0.0, cumulative]
    tail_count = max(1, math.ceil(len(pnl) * 0.01)) if len(pnl) else 0
    worst_tail = np.sort(pnl)[:tail_count] if tail_count else np.asarray([], dtype=float)
    return {
        "trades": len(pnl),
        "net_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "expectancy_per_trade": float(pnl.mean()) if len(pnl) else None,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "average_winning_return": average_win,
        "average_losing_return": average_loss,
        "wins_to_recover_average_loss": (
            abs(average_loss) / average_win
            if average_loss is not None and average_win is not None and average_win > 0
            else None
        ),
        "maximum_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "worst_trade": float(pnl.min()) if len(pnl) else 0.0,
        "worst_one_percent_mean": float(worst_tail.mean()) if len(worst_tail) else 0.0,
    }


def _price_diagnostics(selected: pl.DataFrame) -> dict[str, Any]:
    prices = selected["selected_ask_vwap_5"].to_numpy() if selected.height else np.asarray([])
    bands: dict[str, Any] = {}
    for name, lower, upper in PRICE_BANDS:
        rows = selected.filter(
            pl.col("selected_ask_vwap_5").is_between(lower, upper, closed="left")
        )
        pnl = _execution_metrics(rows, "realized_net_pnl_5")
        bands[name] = {
            "trades": rows.height,
            "accuracy": float(rows["correct"].mean()) if rows.height else None,
            "expectancy_per_trade": pnl["expectancy_per_trade"],
            "net_pnl": pnl["net_pnl"],
        }
    losses_above = {}
    for threshold in (0.85, 0.90, 0.92):
        rows = selected.filter(
            (pl.col("selected_ask_vwap_5") >= threshold) & ~pl.col("correct")
        )
        losses_above[str(threshold)] = {
            "losses": rows.height,
            "net_pnl": float(rows["realized_net_pnl_5"].sum()) if rows.height else 0.0,
        }
    return {
        "mean_vwap5": float(prices.mean()) if len(prices) else None,
        "median_vwap5": float(np.median(prices)) if len(prices) else None,
        "bands": bands,
        "losses_above": losses_above,
    }


def _depth_diagnostics(selected: pl.DataFrame) -> dict[str, Any]:
    slopes = selected["vwap10_minus_vwap5"].to_numpy() if selected.height else np.asarray([])
    bands: dict[str, Any] = {}
    for name, lower, upper in DEPTH_BANDS:
        rows = selected.filter(
            pl.col("vwap10_minus_vwap5").is_between(lower, upper, closed="left")
        )
        pnl = _execution_metrics(rows, "realized_net_pnl_5")
        bands[name] = {
            "trades": rows.height,
            "accuracy": float(rows["correct"].mean()) if rows.height else None,
            "expectancy_per_trade": pnl["expectancy_per_trade"],
            "net_pnl": pnl["net_pnl"],
        }
    return {
        "mean_vwap10_minus_vwap5": float(slopes.mean()) if len(slopes) else None,
        "median_vwap10_minus_vwap5": float(np.median(slopes)) if len(slopes) else None,
        "bands": bands,
    }


def _fold_improves(
    challenger: dict[str, Any],
    control: dict[str, Any],
    config: ChampionVwapBenchmarkConfig,
) -> bool:
    challenger_accuracy = challenger["accuracy"]
    control_accuracy = control["accuracy"]
    return bool(
        challenger["trade_coverage"] >= config.promotion.minimum_champion_coverage
        and challenger_accuracy is not None
        and control_accuracy is not None
        and challenger_accuracy >= control_accuracy - config.promotion.accuracy_tolerance
        and challenger["five_share"]["net_pnl_per_eligible_market"]
        > control["five_share"]["net_pnl_per_eligible_market"]
    )


def _validate_champion_payload(
    model: dict[str, Any],
    manifest: dict[str, Any],
    config: ChampionVwapBenchmarkConfig,
) -> None:
    if model["model_key"] != config.champion_model_key:
        raise RuntimeError("champion model key mismatch")
    if manifest["model_key"] != config.champion_model_key:
        raise RuntimeError("champion manifest model key mismatch")
    if manifest["model_sha256"] != config.champion_model_sha256:
        raise RuntimeError("champion manifest model hash mismatch")
    if manifest["feature_schema_sha256"] != config.champion_feature_schema_sha256:
        raise RuntimeError("champion feature schema hash mismatch")
    decision = model["decision"]
    policy = model["prediction_policy"]
    if not math.isclose(
        float(decision["confidence_threshold"]), config.model.confidence_threshold
    ):
        raise RuntimeError("champion confidence threshold changed")
    timing = (
        int(policy["minimum_seconds_after_open"]),
        int(policy["maximum_seconds_after_open"]),
        int(policy["cadence_seconds"]),
    )
    expected = (
        config.model.minimum_seconds_after_open,
        config.model.maximum_seconds_after_open,
        config.model.cadence_seconds,
    )
    if policy["type"] != "first_confidence_crossing" or timing != expected:
        raise RuntimeError("champion first-crossing policy changed")


def _ece(probability: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    if len(probability) == 0:
        return 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (probability >= lower) & (
            probability <= upper if index == bins - 1 else probability < upper
        )
        if mask.any():
            total += float(mask.mean()) * abs(float(probability[mask].mean() - labels[mask].mean()))
    return total


def _finite_matrix(frame: pl.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError("missing calibration columns: " + ", ".join(missing))
    matrix = frame.select(*names).to_numpy().astype(np.float64)
    if matrix.ndim != 2 or len(matrix) != frame.height or not np.isfinite(matrix).all():
        raise ValueError("calibration inputs must be complete and finite; imputation is disabled")
    return matrix


def _range(frame: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    selected = frame.filter(
        (pl.col("window_start") >= start) & (pl.col("window_start") < end)
    )
    if selected.is_empty():
        raise RuntimeError(f"configured cohort is empty: {start.isoformat()} to {end.isoformat()}")
    return selected


def _cohort(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "start": frame["window_start"].min().isoformat(),
        "end": frame["window_start"].max().isoformat(),
        "up_outcomes": int(frame["label_up"].sum()),
        "down_outcomes": int(frame.height - frame["label_up"].sum()),
        "key_sha256": _key_sha256(frame),
    }


def _key_sha256(frame: pl.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.select("market_id", "observed_at").sort("market_id").iter_rows():
        digest.update(f"{row[0]}|{row[1].isoformat()}\n".encode())
    return digest.hexdigest()


def _check(
    name: str,
    observed: float,
    required: float,
    operator: str,
) -> dict[str, Any]:
    passed = (
        observed > required
        if operator == ">"
        else observed < required
        if operator == "<"
        else observed >= required
    )
    return {
        "name": name,
        "observed": observed,
        "required": required,
        "operator": operator,
        "passed": bool(passed),
    }


def _comparable_profit_factor(metrics: dict[str, Any]) -> float:
    value = metrics["profit_factor"]
    return float(value) if value is not None else (math.inf if metrics["gross_profit"] > 0 else 0.0)


def _comparable_recovery(metrics: dict[str, Any]) -> float:
    value = metrics["wins_to_recover_average_loss"]
    return float(value) if value is not None else (0.0 if metrics["gross_loss"] == 0 else math.inf)


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    clipped = np.clip(logit, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _calibrator_model_key(base_model_key: str, candidate: str) -> str:
    suffix = "vwap5-calibration-v1" if candidate == VWAP5 else "vwap5-vwap10-depth-calibration-v1"
    return f"{base_model_key}-{suffix}"


def _report_html(result: dict[str, Any]) -> str:
    rows = []
    for candidate in CANDIDATES:
        metrics = result["holdout"][candidate]
        five = metrics["five_share"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(candidate)}</td>"
            f"<td>{metrics['trades']}</td>"
            f"<td>{metrics['trade_coverage']:.1%}</td>"
            f"<td>{_fmt(metrics['accuracy'])}</td>"
            f"<td>{_fmt(metrics['expected_calibration_error'])}</td>"
            f"<td>{_fmt(five['net_pnl'])}</td>"
            f"<td>{_fmt(five['net_pnl_per_eligible_market'])}</td>"
            f"<td>{_fmt(five['profit_factor'])}</td>"
            f"<td>{_fmt(five['wins_to_recover_average_loss'])}</td>"
            f"<td>{_fmt(five['maximum_drawdown'])}</td>"
            f"<td>{_fmt(five['worst_one_percent_mean'])}</td>"
            "</tr>"
        )
    checks = []
    for candidate in CHALLENGERS:
        for check in result["promotion"]["candidates"][candidate]["checks"]:
            checks.append(
                "<tr>"
                f"<td>{html.escape(candidate)}</td>"
                f"<td>{html.escape(check['name'])}</td>"
                f"<td>{'yes' if check['passed'] else 'no'}</td>"
                "</tr>"
            )
    selected = result["promotion"]["selected_candidate"] or "none; frozen champion retained"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Champion VWAP calibration benchmark</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:1200px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:.45rem;text-align:right}}th:first-child,td:first-child,td:nth-child(2){{text-align:left}}code{{background:#eee;padding:.15rem}}</style></head>
<body><h1>Frozen champion VWAP calibration benchmark</h1>
<p>Source process: <code>{html.escape(result['source_process_id'])}</code><br>
Champion: <code>{html.escape(result['champion']['model_key'])}</code><br>
Promotion result: <strong>{html.escape(selected)}</strong></p>
<p>The champion classifier and direction are frozen. VWAP10 is used only as depth context; ten-share economics remain separate.</p>
<h2>Final chronological holdout</h2>
<table><thead><tr><th>Candidate</th><th>Trades</th><th>Coverage</th><th>Accuracy</th><th>ECE</th><th>5-share PnL</th><th>PnL / eligible market</th><th>Profit factor</th><th>Wins / avg loss</th><th>Max drawdown</th><th>Worst 1% mean</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Promotion checks</h2><table><thead><tr><th>Candidate</th><th>Requirement</th><th>Passed</th></tr></thead><tbody>{''.join(checks)}</tbody></table>
<p>Full fold, price-band, depth-band, loss-threshold, and calibrator details are in <code>benchmark.json</code>.</p>
</body></html>"""


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return html.escape(str(value))
