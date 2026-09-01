"""Seven-candidate tournament with causal Kraken L2 and hybrid payoff admission."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from . import middle_strategy_tournament as mst
from .core_extract import file_sha256
from .kraken_l2_training_data import (
    KRAKEN_L2_FEATURES,
    attach_kraken_l2,
    build_kraken_l2_features,
)
from .multivenue_early_entry_data import KEY_COLUMNS, load_data_config

SCHEMA_VERSION = "btc-hybrid-payoff-admission-tournament-v1"
ARTIFACT_SCHEMA_VERSION = "btc-hybrid-payoff-admission-model-v1"
CANONICAL_MODULE = "btc_directional_model.hybrid_admission_tournament"
if __name__ == "__main__":
    sys.modules[CANONICAL_MODULE] = sys.modules[__name__]


@dataclass(frozen=True)
class HybridAdmissionModel:
    feature_names: tuple[str, ...]
    profitable_classifier: HistGradientBoostingClassifier
    stress_edge_regressor: HistGradientBoostingRegressor
    lower_bound_penalties: dict[str, float]
    global_lower_bound_penalty: float
    stress_slippage_per_share: float
    uses_l2: bool


HybridAdmissionModel.__module__ = CANONICAL_MODULE

NON_L2_ADMISSION_FEATURES = (
    "probability",
    "selected_probability",
    "expected_edge",
    "share_cost",
    "fee_per_share",
    "seconds_elapsed",
    "probability_change_5s",
    "probability_change_15s",
    "probability_instability_30s",
    "candidate_probability_std",
    "candidate_probability_range",
    "candidate_direction_agreement",
    "btc_path_from_window_open_bps",
    "btc_return_5s_bps",
    "btc_return_30s_bps",
    "btc_realized_volatility_30s_bps",
    "btc_boundary_terminal_volatility_z",
    "chainlink_ref_boundary_gap_bps",
    "chainlink_ref_binance_basis_bps",
    "chainlink_ref_realized_volatility_30s_bps",
    "binance_print_signed_share_30s",
    "binance_oi_change_15m_bps",
    "kraken_return_30s_bps",
    "kraken_print_signed_share_30s",
    "kraken_binance_basis_bps",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _git_revision(package_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=package_root, text=True
    ).strip()


def _hybrid_cache(config: Any) -> Path:
    return config.package_root / config.raw["paths"]["hybrid_cache"]


def build_hybrid_panel(
    config: Any, *, force_l2: bool = False
) -> tuple[pl.DataFrame, dict[str, Any]]:
    cache = _hybrid_cache(config)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "middle-panel.parquet"
    manifest_path = cache / "panel-manifest.json"
    if destination.is_file() and manifest_path.is_file() and not force_l2:
        manifest = json.loads(manifest_path.read_text())
        if manifest["sha256"] != file_sha256(destination):
            raise RuntimeError("hybrid panel changed after checkpoint")
        return pl.read_parquet(destination), manifest

    previous = config.package_root / config.raw["paths"]["input_bridge_cache"]
    base_path = previous / "middle-panel.parquet"
    base_manifest_path = previous / "panel-manifest.json"
    if not base_path.is_file() or not base_manifest_path.is_file():
        raise RuntimeError("source-preserving bridge checkpoint is absent")
    base_manifest = json.loads(base_manifest_path.read_text())
    if base_manifest["sha256"] != file_sha256(base_path):
        raise RuntimeError("source-preserving bridge checkpoint identity changed")
    base = pl.read_parquet(base_path)
    l2_root = Path(config.raw["sources"]["kraken_l2_root"])
    start = datetime.fromisoformat(config.raw["windows"]["kraken_l2_start"])
    end = datetime.fromisoformat(config.raw["windows"]["kraken_l2_end"])
    l2_manifest_path, l2_manifest = build_kraken_l2_features(
        raw_root=l2_root,
        cache=cache,
        start=start,
        end=end,
        force=force_l2,
    )
    panel = attach_kraken_l2(base, l2_manifest)
    panel.write_parquet(destination, compression="zstd", statistics=True)
    feature_groups = dict(base_manifest["feature_groups"])
    feature_groups["kraken_l2"] = [*KRAKEN_L2_FEATURES, "has_kraken_l2"]
    coverage = dict(base_manifest["coverage"])
    coverage["kraken_l2"] = {
        "rows": panel.filter("has_kraken_l2").height,
        "markets": panel.filter("has_kraken_l2")["market_id"].n_unique(),
        "first_observed_at": panel.filter("has_kraken_l2")["observed_at"].min(),
        "last_observed_at": panel.filter("has_kraken_l2")["observed_at"].max(),
    }
    manifest = {
        **base_manifest,
        "schema_version": SCHEMA_VERSION,
        "feature_groups": feature_groups,
        "coverage": coverage,
        "base_panel": {
            "path": str(base_path),
            "sha256": base_manifest["sha256"],
            "manifest_sha256": file_sha256(base_manifest_path),
        },
        "kraken_l2_source": {
            "manifest_path": str(l2_manifest_path),
            "manifest_sha256": file_sha256(l2_manifest_path),
            "representation": l2_manifest["representation"],
        },
        "kraken_l2_included": True,
        "optional_missingness_preserves_rows": True,
        "authentic_only_filter": False,
        "database_mutations": False,
        "new_tables": False,
        "new_ingesters": False,
        "new_sources": False,
        "sha256": file_sha256(destination),
    }
    _write_json(manifest_path, manifest)
    return panel, manifest


def _candidate_contract(config: Any, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mst._configure_roster(config)
    return mst._candidate_contract(config, manifest)


def _prediction_context(predictions: pl.DataFrame) -> pl.DataFrame:
    return predictions.group_by(list(KEY_COLUMNS)).agg(
        pl.col("probability").std().fill_null(0.0).alias("candidate_probability_std"),
        (pl.col("probability").max() - pl.col("probability").min()).alias(
            "candidate_probability_range"
        ),
        (
            pl.max_horizontal(
                (pl.col("probability") >= 0.5).mean(),
                (pl.col("probability") < 0.5).mean(),
            )
        ).alias("candidate_direction_agreement"),
    )


def _time_band_expression(config: Any) -> pl.Expr:
    expression: pl.Expr | None = None
    for name, start, end in mst._calibration_cells(config):
        condition = pl.col("seconds_elapsed").is_between(start, end, closed="both")
        expression = (
            pl.when(condition).then(pl.lit(name))
            if expression is None
            else expression.when(condition).then(pl.lit(name))
        )
    assert expression is not None
    return expression.otherwise(pl.lit("outside")).alias("time_band")


def _admission_frame(
    predictions: pl.DataFrame,
    all_predictions: pl.DataFrame,
    panel: pl.DataFrame,
    config: Any,
) -> pl.DataFrame:
    opportunities = mst._opportunities(predictions, panel, config)
    feature_names = [
        name
        for name in (*NON_L2_ADMISSION_FEATURES[12:], *manifest_l2_features(panel))
        if name in panel.columns
    ]
    context = panel.select(*KEY_COLUMNS, *feature_names)
    frame = opportunities.join(context, on=list(KEY_COLUMNS), how="left", validate="m:1")
    frame = frame.join(
        _prediction_context(all_predictions), on=list(KEY_COLUMNS), how="left", validate="m:1"
    )
    reserve = float(config.raw["execution"]["execution_reserve_per_share"])
    stress = float(config.raw["execution"]["stress_slippage_per_share"])
    correct = ((pl.col("side") == "up") & (pl.col("label_up") == 1)) | (
        (pl.col("side") == "down") & (pl.col("label_up") == 0)
    )
    return (
        frame.sort(["market_id", "seconds_elapsed"])
        .with_columns(
            (pl.col("probability") - pl.col("probability").shift(1).over("market_id"))
            .fill_null(0.0)
            .alias("probability_change_5s"),
            (pl.col("probability") - pl.col("probability").shift(3).over("market_id"))
            .fill_null(0.0)
            .alias("probability_change_15s"),
            pl.col("probability")
            .rolling_std(7)
            .over("market_id")
            .fill_null(0.0)
            .alias("probability_instability_30s"),
            _time_band_expression(config),
            pl.col("share_cost")
            .cut([0.65, 0.80, 0.95], labels=["0", "1", "2", "3"])
            .cast(pl.Int8)
            .alias("price_bucket_index"),
            correct.alias("direction_correct"),
        )
        .with_columns(
            (
                pl.col("direction_correct").cast(pl.Float64)
                - pl.col("share_cost")
                - pl.col("fee_per_share")
                - reserve
                - stress
            ).alias("realized_stress_edge")
        )
    )


def manifest_l2_features(panel: pl.DataFrame) -> tuple[str, ...]:
    return tuple(
        name for name in panel.columns if name.startswith(("spot_l2_", "kraken_l2_"))
    ) + tuple(name for name in ("has_spot_l2", "has_kraken_l2") if name in panel.columns)


def _matrix(frame: pl.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    return np.column_stack(
        [frame[name].cast(pl.Float64).fill_nan(None).fill_null(0.0).to_numpy() for name in features]
    )


def _market_weights(frame: pl.DataFrame) -> np.ndarray:
    counts = frame.group_by("market_id").len().rename({"len": "market_rows"})
    return (
        frame.join(counts, on="market_id", how="left")["market_rows"]
        .cast(pl.Float64)
        .pow(-1)
        .to_numpy()
    )


def _fit_admission(frame: pl.DataFrame, features: tuple[str, ...], seed: int) -> tuple[Any, Any]:
    valid = frame.filter(
        pl.col("share_cost").is_not_null() & pl.col("realized_stress_edge").is_not_null()
    )
    if valid["market_id"].n_unique() < 250:
        raise RuntimeError("hybrid admission model lacks chronological training markets")
    weights = _market_weights(valid)
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=5.0,
        early_stopping=False,
        random_state=seed,
    ).fit(
        _matrix(valid, features),
        (valid["realized_stress_edge"] > 0).cast(pl.Int8),
        sample_weight=weights,
    )
    regressor = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=5.0,
        early_stopping=False,
        random_state=seed + 1,
    ).fit(_matrix(valid, features), valid["realized_stress_edge"], sample_weight=weights)
    return classifier, regressor


def _features(frame: pl.DataFrame, *, uses_l2: bool) -> tuple[str, ...]:
    names = [name for name in NON_L2_ADMISSION_FEATURES if name in frame.columns]
    if uses_l2:
        names.extend(name for name in manifest_l2_features(frame) if name not in names)
    varying = []
    for name in names:
        if frame[name].drop_nulls().n_unique() > 1:
            varying.append(name)
    return tuple(varying)


def _attach_admission(frame: pl.DataFrame, model: HybridAdmissionModel) -> pl.DataFrame:
    matrix = _matrix(frame, model.feature_names)
    probability = model.profitable_classifier.predict_proba(matrix)[:, 1]
    expected = model.stress_edge_regressor.predict(matrix)
    penalties = np.asarray(
        [
            model.lower_bound_penalties.get(f"{band}:{bucket}", model.global_lower_bound_penalty)
            for band, bucket in zip(
                frame["time_band"].to_list(), frame["price_bucket_index"].to_list(), strict=True
            )
        ]
    )
    conditional_loss = (
        frame["share_cost"].fill_null(1.0).to_numpy() + model.stress_slippage_per_share
    )
    return frame.with_columns(
        pl.Series("admission_probability", probability),
        pl.Series("payoff_expected_stress_edge", expected),
        pl.Series("payoff_stress_edge_lower_bound", expected - penalties),
        pl.Series("payoff_expected_shortfall", (1.0 - probability) * conditional_loss),
    )


def fit_admission_oof(
    frame: pl.DataFrame,
    config: Any,
    *,
    uses_l2: bool,
    seed: int,
) -> tuple[HybridAdmissionModel, pl.DataFrame, dict[str, Any]]:
    features = _features(frame, uses_l2=uses_l2)
    fold_names = [row["name"] for row in config.raw["folds"]]
    pieces: list[pl.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for index, fold in enumerate(fold_names[1:], start=1):
        fit = frame.filter(pl.col("fold").is_in(fold_names[:index]))
        validation = frame.filter(pl.col("fold") == fold)
        if fit["market_id"].n_unique() < 250 or validation["market_id"].n_unique() < 40:
            continue
        classifier, regressor = _fit_admission(fit, features, seed + 100 * index)
        temporary = HybridAdmissionModel(
            features,
            classifier,
            regressor,
            {},
            0.0,
            float(config.raw["execution"]["stress_slippage_per_share"]),
            uses_l2,
        )
        scored = _attach_admission(validation, temporary)
        pieces.append(scored)
        diagnostics.append(
            {
                "fold": fold,
                "training_markets": fit["market_id"].n_unique(),
                "validation_markets": validation["market_id"].n_unique(),
            }
        )
    if not pieces:
        raise RuntimeError("hybrid admission OOF produced no chronological folds")
    oof = pl.concat(pieces, how="vertical_relaxed", rechunk=True)
    residual = (
        oof["payoff_expected_stress_edge"].to_numpy() - oof["realized_stress_edge"].to_numpy()
    )
    global_penalty = float(np.quantile(residual, 0.90))
    penalties: dict[str, float] = {}
    for band, _, _ in mst._calibration_cells(config):
        for bucket in range(4):
            cell = oof.filter(
                (pl.col("time_band") == band) & (pl.col("price_bucket_index") == bucket)
            )
            if cell.is_empty():
                penalties[f"{band}:{bucket}"] = global_penalty
                continue
            values = (
                cell["payoff_expected_stress_edge"].to_numpy()
                - cell["realized_stress_edge"].to_numpy()
            )
            cell_penalty = float(np.quantile(values, 0.90))
            markets = cell["market_id"].n_unique()
            shrink = markets / (markets + 200.0)
            penalties[f"{band}:{bucket}"] = shrink * cell_penalty + (1.0 - shrink) * global_penalty
    classifier, regressor = _fit_admission(frame, features, seed + 10_000)
    model = HybridAdmissionModel(
        features,
        classifier,
        regressor,
        penalties,
        global_penalty,
        float(config.raw["execution"]["stress_slippage_per_share"]),
        uses_l2,
    )
    oof_penalties = np.asarray(
        [
            penalties.get(f"{band}:{bucket}", global_penalty)
            for band, bucket in zip(
                oof["time_band"].to_list(), oof["price_bucket_index"].to_list(), strict=True
            )
        ]
    )
    oof = oof.with_columns(
        pl.Series(
            "payoff_stress_edge_lower_bound",
            oof["payoff_expected_stress_edge"].to_numpy() - oof_penalties,
        )
    )
    return (
        model,
        oof,
        {
            "features": list(features),
            "folds": diagnostics,
            "oof_markets": oof["market_id"].n_unique(),
            "global_lower_bound_penalty": global_penalty,
            "empirical_lower_bound_coverage": float(
                (oof["realized_stress_edge"] >= oof["payoff_stress_edge_lower_bound"]).mean()
            ),
        },
    )


def _basic_policy(config: Any) -> dict[str, float]:
    execution = config.raw["execution"]
    return {
        "minimum_edge": float(min(execution["minimum_edges"])),
        "minimum_confidence": float(min(execution["minimum_confidences"])),
        "maximum_share_cost": float(max(execution["maximum_share_costs"])),
        "abstain": False,
    }


def _hybrid_trades(
    frame: pl.DataFrame,
    base_policy: dict[str, float],
    policy: dict[str, float],
    config: Any,
) -> pl.DataFrame:
    if policy.get("abstain", False):
        return mst._select_trades(frame.head(0), base_policy, config)
    eligible = frame.filter(
        (pl.col("admission_probability") >= policy["minimum_admission_probability"])
        & (pl.col("payoff_stress_edge_lower_bound") >= policy["minimum_stress_edge_lower_bound"])
        & (pl.col("payoff_expected_shortfall") <= policy["maximum_expected_shortfall"])
    )
    return mst._select_trades(eligible, base_policy, config)


def select_hybrid_policies(
    frame: pl.DataFrame, config: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = _basic_policy(config)
    selected: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    spec = config.raw["hybrid_admission"]
    for band, start, end in mst._calibration_cells(config):
        part = frame.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both"))
        best: tuple[float, dict[str, float], pl.DataFrame] | None = None
        attempted = 0
        for probability in spec["probability_thresholds"]:
            for lower_bound in spec["lower_bound_thresholds"]:
                for shortfall in spec["maximum_expected_shortfalls"]:
                    attempted += 1
                    policy = {
                        "minimum_admission_probability": float(probability),
                        "minimum_stress_edge_lower_bound": float(lower_bound),
                        "maximum_expected_shortfall": float(shortfall),
                        "abstain": False,
                    }
                    trades = _hybrid_trades(part, base, policy, config)
                    if trades.is_empty():
                        continue
                    fold_pnl = trades.group_by("fold").agg(pl.col("net_pnl").sum())
                    stable = float((fold_pnl["net_pnl"] > 0).mean()) >= 0.50
                    score = mst._policy_score(trades)
                    if (
                        stable
                        and trades["stress_net_pnl"].sum() > 0
                        and score > 0
                        and (best is None or score > best[0])
                    ):
                        best = (score, policy, trades)
        if best is None:
            selected[band] = {**base, "abstain": True}
            evidence[band] = {"attempted": attempted, "abstained": True}
        else:
            score, policy, trades = best
            selected[band] = policy
            evidence[band] = {
                "attempted": attempted,
                "abstained": False,
                "selection_score": score,
                "rolling_oof": mst.economic_metrics(trades, part["market_id"].n_unique()),
            }
    return selected, evidence


def apply_hybrid_policies(
    frame: pl.DataFrame, policies: dict[str, Any], config: Any
) -> pl.DataFrame:
    pieces = []
    base = _basic_policy(config)
    for band, start, end in mst._calibration_cells(config):
        part = frame.filter(pl.col("seconds_elapsed").is_between(start, end, closed="both"))
        pieces.append(_hybrid_trades(part, base, policies[band], config))
    output = pl.concat(pieces, how="diagonal_relaxed")
    if output.is_empty():
        return output
    return (
        output.sort(["market_id", "seconds_elapsed"])
        .group_by("market_id", maintain_order=True)
        .first()
    )


def _validate_artifact(config: Any, artifact: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(config.package_root / "src")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import joblib,sys; x=joblib.load(sys.argv[1]); assert len(x['base_models'])==5; assert len(x['admission_models'])==14",
            str(artifact),
        ],
        check=True,
        cwd=config.package_root,
        env=environment,
    )


def _evaluate_modes(
    *,
    predictions: pl.DataFrame,
    all_predictions: pl.DataFrame,
    panel: pl.DataFrame,
    programmatic: dict[str, Any],
    admission_models: dict[str, dict[str, HybridAdmissionModel]],
    hybrid_policies: dict[str, dict[str, Any]],
    config: Any,
) -> tuple[dict[str, Any], pl.DataFrame]:
    metrics: dict[str, Any] = {}
    pieces: list[pl.DataFrame] = []
    total = panel["market_id"].n_unique()
    for candidate in mst.ALL_NAMES:
        candidate_predictions = predictions.filter(pl.col("candidate") == candidate)
        admission = _admission_frame(candidate_predictions, all_predictions, panel, config)
        modes: dict[str, Any] = {}
        no_veto = mst._select_trades(admission, _basic_policy(config), config)
        current = mst._apply_candidate_policy(admission, programmatic[candidate], candidate, config)
        mode_trades = {"veto_disabled": no_veto, "programmatic": current}
        for mode in ("hybrid_no_l2", "hybrid_dual_l2"):
            scored = _attach_admission(admission, admission_models[candidate][mode])
            mode_trades[mode] = apply_hybrid_policies(
                scored, hybrid_policies[candidate][mode], config
            )
        for mode, trades in mode_trades.items():
            modes[mode] = mst.economic_metrics(trades, total)
            if not trades.is_empty():
                pieces.append(
                    trades.with_columns(
                        pl.lit(candidate).alias("candidate"), pl.lit(mode).alias("veto_mode")
                    )
                )
            late = trades.filter(pl.col("seconds_elapsed").is_between(150, 180, closed="both"))
            modes[mode]["entries_150_180_only"] = mst.economic_metrics(late, total)
        metrics[candidate] = modes
    return metrics, pl.concat(pieces, how="diagonal_relaxed") if pieces else pl.DataFrame()


def _report(metrics: dict[str, Any]) -> str:
    def fmt(value: Any, digits: int = 3) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Hybrid Payoff-Admission Tournament",
        "",
        f"Run: `{metrics['run_id']}`",
        f"Qualification: **{metrics['qualification_status']}**",
        "",
        "## Heldout August 21–26 summary",
        "",
        "| Candidate | Veto | PnL | Stress PnL | PF | Coverage | W | L | W/L | Recovery wins/loss | Avg entry | Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in metrics["candidate_names"]:
        brier = metrics["heldout_predictive"][candidate]["brier_score"]
        for mode, row in metrics["heldout_economic"][candidate].items():
            lines.append(
                f"| {candidate} | {mode} | {row['net_pnl']:.2f} | {row['stress_net_pnl']:.2f} | {fmt(row['profit_factor'])} | {row['market_coverage']:.2%} | {row['winning_trades']} | {row['losing_trades']} | {fmt(row['win_loss_ratio'])} | {fmt(row['loss_recovery_wins'])} | {fmt(row['average_entry_second'], 1)} | {fmt(brier, 4)} |"
            )
    lines.extend(
        (
            "",
            "## Integrity",
            "",
            "- Predictive fitting, nested admission fitting, threshold selection, and heldout testing are chronological and market-disjoint.",
            "- Kraken L2 is represented as causal incremental update flow, not as a reconstructed full order book.",
            "- Optional L2 gaps preserve all markets and are represented by missingness/availability features.",
            "- TWAP and RefPrice settlement values are supervision only and are absent from inference and admission features.",
            "- No database writes, tables, schemas, ingesters, data sources, runtime exports, deployments, or image rebuilds occurred.",
        )
    )
    return "\n".join(lines) + "\n"


def train_tournament(config: Any, *, force_l2: bool = False) -> Path:
    mst._configure_roster(config)
    panel, panel_manifest = build_hybrid_panel(config, force_l2=force_l2)
    contracts = _candidate_contract(config, panel_manifest)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.results / run_id
    ledgers = run_dir / "ledgers"
    ledgers.mkdir(parents=True, exist_ok=False)
    fit = panel.filter(pl.col("window_start") < config.fit_end)
    heldout = panel.filter(
        pl.col("window_start").is_between(config.sealed_start, config.sealed_end, closed="left")
    )
    if set(fit["market_id"].unique()) & set(heldout["market_id"].unique()):
        raise RuntimeError("predictive fit and heldout markets overlap")
    split = {
        "source_start": config.source_start.isoformat(),
        "fit_end_exclusive": config.fit_end.isoformat(),
        "heldout_start_inclusive": config.sealed_start.isoformat(),
        "heldout_end_exclusive": config.sealed_end.isoformat(),
        "folds": config.raw["folds"],
        "predictive_oof_only_for_admission": True,
        "nested_chronological_admission_oof": True,
        "heldout_excluded_from_model_and_policy_selection": True,
        "market_disjoint": True,
    }
    _write_json(run_dir / "split-manifest.json", split)

    base_oof = mst._base_oof(fit, contracts, config)
    oof, wide_oof = mst._all_oof(base_oof, config)
    oof.write_parquet(
        ledgers / "candidate-oof-predictions.parquet", compression="zstd", statistics=True
    )
    predictive = {
        name: mst.predictive_metrics(oof.filter(pl.col("candidate") == name))
        for name in mst.ALL_NAMES
    }

    seed = int(config.raw["training"]["random_seed"])
    final_models = {}
    for index, name in enumerate(mst.BASE_NAMES):
        final_models[name] = mst._fit_candidate_tree(
            fit,
            tuple(contracts[name]["features"]),
            config,
            seed + 10_000 + index,
            target_kind=contracts[name]["kind"],
        )
    final_calibrator = mst._fit_ensemble_calibrator(wide_oof, seed + 20_000)
    joblib.dump(
        {"base_models": final_models, "ensemble_calibrator": final_calibrator},
        run_dir / "predictive-model-checkpoint.joblib",
        compress=3,
    )

    programmatic: dict[str, Any] = {}
    programmatic_evidence: dict[str, Any] = {}
    admission_models: dict[str, dict[str, HybridAdmissionModel]] = {}
    hybrid_policies: dict[str, dict[str, Any]] = {}
    admission_evidence: dict[str, Any] = {}
    for index, candidate in enumerate(mst.ALL_NAMES):
        candidate_oof = oof.filter(pl.col("candidate") == candidate)
        frame = _admission_frame(candidate_oof, oof, fit, config)
        programmatic[candidate], programmatic_evidence[candidate] = mst._select_cell_policies(
            frame, config
        )
        admission_models[candidate] = {}
        hybrid_policies[candidate] = {}
        admission_evidence[candidate] = {}
        for offset, (mode, uses_l2) in enumerate(
            (("hybrid_no_l2", False), ("hybrid_dual_l2", True))
        ):
            model, admission_oof, evidence = fit_admission_oof(
                frame, config, uses_l2=uses_l2, seed=seed + 30_000 + 100 * index + offset
            )
            admission_models[candidate][mode] = model
            policies, policy_evidence = select_hybrid_policies(admission_oof, config)
            hybrid_policies[candidate][mode] = policies
            admission_evidence[candidate][mode] = {**evidence, "policy": policy_evidence}
        joblib.dump(
            admission_models,
            run_dir / "admission-model-checkpoint.joblib",
            compress=3,
        )
    selection_frozen_at = datetime.now(UTC).isoformat()
    _write_json(
        run_dir / "selection-freeze.json",
        {
            "frozen_at": selection_frozen_at,
            "programmatic": programmatic,
            "hybrid": hybrid_policies,
            "admission_evidence": admission_evidence,
            "heldout_metrics_accessed": False,
        },
    )

    heldout_predictions = mst._predict_frozen_candidates(
        heldout, final_models, final_calibrator, "heldout_20260821_20260827"
    )
    heldout_predictions.write_parquet(
        ledgers / "heldout-predictions.parquet", compression="zstd", statistics=True
    )
    heldout_predictive = {
        name: mst.predictive_metrics(heldout_predictions.filter(pl.col("candidate") == name))
        for name in mst.ALL_NAMES
    }
    heldout_economic, heldout_trades = _evaluate_modes(
        predictions=heldout_predictions,
        all_predictions=heldout_predictions,
        panel=heldout,
        programmatic=programmatic,
        admission_models=admission_models,
        hybrid_policies=hybrid_policies,
        config=config,
    )
    heldout_trades.write_parquet(
        ledgers / "heldout-trades.parquet", compression="zstd", statistics=True
    )

    producing_commit = _git_revision(config.package_root)
    artifact = run_dir / "tournament.joblib"
    joblib.dump(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "producing_commit": producing_commit,
            "candidate_contract": contracts,
            "base_models": final_models,
            "ensemble_calibrator": final_calibrator,
            "admission_models": {
                f"{candidate}:{mode}": model
                for candidate, modes in admission_models.items()
                for mode, model in modes.items()
            },
            "programmatic_policies": programmatic,
            "hybrid_policies": hybrid_policies,
            "deployment_status": "not_deployed",
        },
        artifact,
        compress=3,
    )
    _validate_artifact(config, artifact)
    artifact_sha = file_sha256(artifact)
    (run_dir / "tournament.sha256").write_text(f"{artifact_sha}  tournament.joblib\n")
    positive = [
        (candidate, mode)
        for candidate, modes in heldout_economic.items()
        for mode, row in modes.items()
        if row["net_pnl"] > 0 and row["stress_net_pnl"] > 0 and (row["profit_factor"] or 0) > 1
    ]
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "producing_commit": producing_commit,
        "candidate_names": list(mst.ALL_NAMES),
        "candidate_count_trained": 7,
        "learned_base_model_count": 5,
        "derived_ensemble_count": 2,
        "admission_model_count": 14,
        "selection_frozen_at": selection_frozen_at,
        "oof_predictive": predictive,
        "heldout_predictive": heldout_predictive,
        "heldout_economic": heldout_economic,
        "positive_expectancy_modes": positive,
        "qualification_status": "trained_evaluated_not_deployed"
        if positive
        else "trained_evaluated_not_promoted_negative_expectancy",
        "source_panel": panel_manifest,
        "split_manifest": split,
        "artifact_sha256": artifact_sha,
        "frozen_comparator_references": mst._reference_manifests(config),
        "integrity": {
            "passed": True,
            "market_disjoint": True,
            "nested_chronological_admission_oof": True,
            "artifact_round_trip_load": True,
            "full_history_retained": True,
            "optional_missingness_preserves_rows": True,
            "twap_inference_feature": False,
            "kraken_l2_included": True,
            "database_mutations": False,
            "new_tables": False,
            "new_schemas": False,
            "new_ingesters": False,
            "new_sources": False,
            "images_rebuilt": False,
            "runtime_exported": False,
            "deployed": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "sklearn": sklearn.__version__,
        },
        "limitations": [
            "August 21-26 is chronologically held out but was observed during earlier research, so it is not epistemically fresh.",
            "Kraken L2 archive coverage ends during August 19 and therefore provides no direct L2 observations in the August 21-26 heldout block.",
            "The Kraken files contain incremental updates without periodic snapshots; features represent update flow rather than exact full-book depth.",
            "Projected PnL assumes recorded Polymarket ask VWAP was fillable and does not model queue position.",
        ],
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "candidate-contract.json", contracts)
    _write_json(run_dir / "source-manifest.json", panel_manifest)
    (run_dir / "report.md").write_text(_report(metrics))
    _write_json(
        run_dir / "completion.json",
        {
            "run_id": run_id,
            "completed": True,
            "artifact_sha256": artifact_sha,
            "metrics_sha256": file_sha256(run_dir / "metrics.json"),
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force-l2", action="store_true")
    args = parser.parse_args()
    result = train_tournament(load_data_config(args.config), force_l2=args.force_l2)
    print(result)


if __name__ == "__main__":
    main()
