"""Train and export causal, process-compatible UMR risk packages.

The fit cohort is the established historical tournament panel. Thresholds are
calibrated on the sealed compatibility cohort before the pristine test cohort;
the same fitted model is replayed against every deployed strategy and time bucket.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

from .runtime_export import export_tree

RUNTIME_FEATURES = (
    "seconds_elapsed",
    "selected_probability",
    "confidence",
    "share_cost",
    "fee_per_share",
    "expected_edge",
    "side_is_up",
    "pm_vwap5_overround",
    "pm_selected_imbalance",
    "pm_selected_book_age_seconds",
)
MANIFEST_FEATURES = RUNTIME_FEATURES + ("chainlink_gap_bps", "realized_volatility")
BUCKETS = {
    "60_89": (60, 89),
    "90_119": (90, 119),
    "120_149": (120, 149),
    "150_179": (150, 179),
    "180_240": (180, 240),
}
CELLS = (
    (
        "candidate-loss-official-vwap-60-89-c95",
        "btc-5m-official-vwap-admission-umr-20260902",
        "60_89",
        0.95,
    ),
    (
        "candidate-loss-temporal-consensus-60-89-c70",
        "btc-5m-official-temporal-consensus-umr-20260902-confidence-075",
        "60_89",
        0.70,
    ),
    (
        "candidate-loss-specialist-distilled-90-119-c70",
        "btc-5m-specialist-distilled-fair-value-paper-20260823-v1",
        "90_119",
        0.70,
    ),
    (
        "candidate-loss-extended-specialist-120-149-c85",
        "btc-5m-extended-specialist-official-umr-20260902-confidence-070",
        "120_149",
        0.85,
    ),
    (
        "candidate-loss-extended-specialist-180-240-c85",
        "btc-5m-extended-specialist-official-umr-20260902-confidence-070",
        "180_240",
        0.85,
    ),
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def prepare(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.col("side") == "UP")
        .then(pl.col("pm_up_book_age_seconds"))
        .otherwise(pl.col("pm_down_book_age_seconds"))
        .alias("pm_selected_book_age_seconds"),
        pl.col("pm_depth_imbalance").alias("pm_selected_imbalance"),
    )


def matrix(frame: pl.DataFrame) -> np.ndarray:
    return frame.select(RUNTIME_FEATURES).to_numpy().astype(np.float64, copy=False)


def selected_trades(
    frame: pl.DataFrame, scores: np.ndarray | None, threshold: float | None, bucket: str | None
) -> pl.DataFrame:
    work = (
        frame.with_columns(pl.Series("_risk", scores))
        if scores is not None
        else frame.with_columns(pl.lit(0.0).alias("_risk"))
    )
    if threshold is not None and bucket is not None:
        lo, hi = BUCKETS[bucket]
        work = work.filter(
            ~(pl.col("seconds_elapsed").is_between(lo, hi) & (pl.col("_risk") >= threshold))
        )
    return (
        work.sort(["champion", "market_id", "observed_at"])
        .group_by(["champion", "market_id"], maintain_order=True)
        .first()
    )


def metrics(base: pl.DataFrame, kept: pl.DataFrame) -> dict[str, float | int | None]:
    pnl = kept["net_pnl"].to_numpy()
    base_pnl = base["net_pnl"].to_numpy()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    equity = np.cumsum(pnl)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])[1:]
    base_by_market = {
        r["market_id"]: r["net_pnl"]
        for r in base.select("market_id", "net_pnl").iter_rows(named=True)
    }
    kept_ids = set(kept["market_id"].to_list())
    blocked = np.array([v for k, v in base_by_market.items() if k not in kept_ids])
    bad = int(np.sum(base_pnl < 0))
    good = int(np.sum(base_pnl > 0))
    bad_blocked = int(np.sum(blocked < 0))
    good_blocked = int(np.sum(blocked > 0))
    return {
        "trades": len(pnl),
        "wins": len(wins),
        "losses": len(losses),
        "pnl": float(pnl.sum()),
        "delta_pnl": float(pnl.sum() - base_pnl.sum()),
        "coverage": len(pnl) / len(base_pnl) if len(base_pnl) else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum())
        if len(losses) and losses.sum()
        else None,
        "recovery_wins_per_loss": float((-losses.mean()) / wins.mean())
        if len(losses) and len(wins)
        else None,
        "max_drawdown": float(np.max(peaks - equity)) if len(equity) else 0.0,
        "average_entry_second": float(kept["seconds_elapsed"].mean()) if len(kept) else None,
        "bad_trades_blocked_pct": bad_blocked / bad if bad else 0.0,
        "good_trades_blocked_pct": good_blocked / good if good else 0.0,
        "risk_alignment_ratio": (bad_blocked / bad) / (good_blocked / good)
        if good_blocked and bad
        else None,
    }


def run(training_path: Path, strategy_path: Path, output: Path, runtime_root: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    train = prepare(pl.read_parquet(training_path))
    strategy = prepare(pl.read_parquet(strategy_path))
    calibration = strategy.filter(pl.col("observed_at") < datetime(2026, 8, 21, tzinfo=UTC))
    test = strategy.filter(pl.col("observed_at") >= datetime(2026, 8, 21, tzinfo=UTC))
    cohorts = [set(frame["market_id"].to_list()) for frame in (train, calibration, test)]
    if cohorts[0] & cohorts[1] or cohorts[0] & cohorts[2] or cohorts[1] & cohorts[2]:
        raise RuntimeError("market leakage across fit/calibration/test cohorts")
    estimator = HistGradientBoostingClassifier(
        learning_rate=0.055,
        max_iter=80,
        max_leaf_nodes=15,
        min_samples_leaf=120,
        l2_regularization=2.0,
        class_weight="balanced",
        early_stopping=True,
        random_state=20260911,
    )
    y = (train["net_pnl"].to_numpy() < 0).astype(np.int8)
    weights = np.maximum(np.abs(train["net_pnl"].to_numpy()), 0.05)
    estimator.fit(matrix(train), y, sample_weight=weights)
    trees = [export_tree(p[0], len(RUNTIME_FEATURES)) for p in estimator._predictors]
    submodel = {
        "feature_indices": list(range(len(RUNTIME_FEATURES))),
        "baseline": float(estimator._baseline_prediction[0, 0]),
        "output": "probability",
        "trees": trees,
    }
    cal_scores = estimator.predict_proba(matrix(calibration))[:, 1]
    test_scores = estimator.predict_proba(matrix(test))[:, 1]
    rows = []
    packages = []
    for suffix, champion, bucket, target_coverage in CELLS:
        mask = (calibration["champion"].to_numpy() == champion) & (
            calibration["time_bucket"].to_numpy() == bucket
        )
        threshold = (
            float(np.quantile(cal_scores[mask], target_coverage))
            if mask.any()
            else float(np.quantile(cal_scores, target_coverage))
        )
        key = f"btc-5m-risk-{suffix}-20260911"
        package = runtime_root / key
        package.mkdir(parents=True, exist_ok=True)
        lo, hi = BUCKETS[bucket]
        model_bytes = _json_bytes(
            {
                "model": submodel,
                "threshold": threshold,
                "qualified_start_second": lo,
                "qualified_end_second": hi,
            }
        )
        artifact_sha = _sha(model_bytes)
        (package / "risk-model.json").write_bytes(model_bytes)
        manifest = {
            "version": "capitonic-risk-strategy-v1",
            "model_key": key,
            "model_sha256": artifact_sha,
            "feature_schema_version": "btc-risk-candidate-v1",
            "feature_names": list(MANIFEST_FEATURES),
            "threshold": threshold,
            "qualified_start_second": lo,
            "qualified_end_second": hi,
            "deployment": {
                "scope": "paper_only",
                "production_qualified": False,
                "live_capital_allowed": False,
            },
            "provenance": {
                "training_path": str(training_path),
                "strategy_path": str(strategy_path),
                "fit_end": "2026-08-10",
                "calibration_end": "2026-08-21",
                "test_start": "2026-08-21",
                "seed": 20260911,
                "strategy": champion,
                "target_coverage": target_coverage,
            },
        }
        (package / "risk-manifest.json").write_bytes(_json_bytes(manifest))
        golden_count = min(8, len(test))
        golden = [
            {
                "features": [float(v) if np.isfinite(v) else None for v in row],
                "expected_score": float(score),
            }
            for row, score in zip(
                matrix(test)[:golden_count], test_scores[:golden_count], strict=True
            )
        ]
        (package / "golden-predictions.json").write_bytes(_json_bytes(golden))
        packages.append(
            {
                "model_key": key,
                "artifact_sha256": artifact_sha,
                "champion": champion,
                "bucket": bucket,
                "threshold": threshold,
            }
        )
        for candidate in sorted(test["champion"].unique().to_list()):
            for candidate_bucket in BUCKETS:
                subset = test.with_columns(pl.Series("_score", test_scores)).filter(
                    (pl.col("champion") == candidate) & (pl.col("time_bucket") == candidate_bucket)
                )
                b = selected_trades(subset, None, None, None)
                k = selected_trades(subset, subset["_score"].to_numpy(), threshold, bucket)
                if len(b):
                    rows.append(
                        {
                            "risk_model": key,
                            "strategy_model": candidate,
                            "time_bucket": candidate_bucket,
                            **metrics(b, k),
                        }
                    )
    pl.DataFrame(rows).write_parquet(output / "all-results.parquet")
    (output / "all-results.json").write_bytes(_json_bytes(rows))
    (output / "packages.json").write_bytes(_json_bytes(packages))
    report = {
        "run_id": output.name,
        "fit_rows": len(train),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "fit_range": [str(train["observed_at"].min()), str(train["observed_at"].max())],
        "calibration_range": [
            str(calibration["observed_at"].min()),
            str(calibration["observed_at"].max()),
        ],
        "test_range": [str(test["observed_at"].min()), str(test["observed_at"].max())],
        "training_panel_sha256": _file_sha(training_path),
        "strategy_panel_sha256": _file_sha(strategy_path),
        "market_overlap_counts": [
            len(cohorts[0] & cohorts[1]),
            len(cohorts[0] & cohorts[2]),
            len(cohorts[1] & cohorts[2]),
        ],
        "features": list(RUNTIME_FEATURES),
        "packages": packages,
        "results": rows,
    }
    (output / "report.json").write_bytes(_json_bytes(report))
    selected = []
    for package in packages:
        selected.extend(
            row
            for row in rows
            if row["risk_model"] == package["model_key"]
            and row["strategy_model"] == package["champion"]
            and row["time_bucket"] == package["bucket"]
        )
    lines = [
        "# Runtime risk tournament",
        "",
        f"Fit: {len(train):,} rows · calibration: {len(calibration):,} rows · pristine test: {len(test):,} rows",
        "",
        "| Risk package | Strategy model | Bucket | Trades | W/L | PnL | Delta PnL | Coverage | PF | Recovery | Max DD | Avg entry | Bad blocked | Good blocked | Alignment |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        pf = "—" if row["profit_factor"] is None else f"{row['profit_factor']:.3f}"
        alignment = (
            "—" if row["risk_alignment_ratio"] is None else f"{row['risk_alignment_ratio']:.2f}x"
        )
        lines.append(
            f"| {row['risk_model']} | {row['strategy_model']} | {row['time_bucket'].replace('_', '–')}s | {row['trades']} | {row['wins']}/{row['losses']} | ${row['pnl']:+.2f} | ${row['delta_pnl']:+.2f} | {row['coverage']:.1%} | {pf} | {row['recovery_wins_per_loss']:.3f} | ${row['max_drawdown']:.2f} | {row['average_entry_second']:.1f}s | {row['bad_trades_blocked_pct']:.1%} | {row['good_trades_blocked_pct']:.1%} | {alignment} |"
        )
    lines.extend(
        [
            "",
            "PnL, profit factor, recovery, drawdown and coverage describe the trading strategy after applying the named risk package. Delta PnL is the change from that strategy/time-bucket no-risk baseline. All 150 risk-package × strategy-model × time-bucket rows are in `all-results.parquet` and `all-results.json`.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))
    (output / "COMPLETE").write_text("complete\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--training-panel", type=Path, required=True)
    p.add_argument("--strategy-panel", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path, required=True)
    a = p.parse_args()
    run(a.training_panel, a.strategy_panel, a.output, a.runtime_root)


if __name__ == "__main__":
    main()
