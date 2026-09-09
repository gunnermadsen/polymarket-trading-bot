"""Train and replay a pooled candidate-loss risk model against frozen champions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import tomllib
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCHEMA_VERSION = "candidate_loss_risk_tournament_v1"
BUCKETS = ("60_89", "90_119", "120_149", "150_180")

ECONOMIC_FEATURES = (
    "seconds_elapsed",
    "selected_probability",
    "confidence",
    "share_cost",
    "fee_per_share",
    "expected_edge",
    "pm_vwap5_overround",
    "pm_vwap50_overround",
    "pm_vwap200_overround",
    "pm_depth_imbalance",
    "pm_up_book_age_seconds",
    "pm_down_book_age_seconds",
)
MARKET_FEATURES = (
    "btc_path_from_window_open_bps",
    "btc_return_5s_bps",
    "btc_return_15s_bps",
    "btc_return_30s_bps",
    "btc_return_60s_bps",
    "btc_realized_volatility_30s_bps",
    "btc_realized_volatility_60s_bps",
    "oracle_gap_to_opening_boundary_bps",
    "binance_oracle_basis_bps",
    "chainlink_ref_return_30s_bps",
    "chainlink_ref_spread_bps",
    "spot_l2_spread_bps",
    "spot_l2_bid_depth_20_log",
    "spot_l2_ask_depth_20_log",
    "kraken_return_30s_bps",
    "kraken_l2_age_seconds",
)
HISTORY_FEATURES = (
    "history_loss_streak",
    "history_win_rate_5",
    "history_win_rate_10",
    "history_win_rate_25",
    "history_net_pnl_25",
    "history_stress_pnl_25",
    "history_brier_25",
    "history_expected_gap_25",
    "history_side_win_rate_25",
    "history_bucket_win_rate_25",
)

CHAMPIONS = {
    "middle_specialist_refit": ("full", "hybrid_full_vwap_no_l2", "150_180", 0.0874),
    "middle_q5_admission": ("full", "hybrid_full_vwap_dual_l2", "150_180", 0.0272),
    "crossvenue_middle_specialist": ("full", "hybrid_full_vwap_dual_l2", "150_180", 0.0349),
    "middle_agreement_ensemble": ("full", "hybrid_full_vwap_dual_l2", "150_180", 0.0465),
    "price_time_calibrated_middle_ensemble": ("full", "hybrid_full_vwap_no_l2", "150_180", 0.0997),
    "extended_specialist_official": ("extended", "programmatic", "60_89", 0.0550),
    "bridge_aware_specialist": ("extended", "programmatic", "60_89", 0.2170),
    "official_vwap_admission": ("admission", None, "60_89", 0.2392),
    "official_temporal_consensus": ("robust", None, "60_89", 0.0230),
    "official_high_precision_loss_veto": ("robust", None, "60_89", 0.0142),
}
PUBLISHED_BASELINES = {
    "middle_specialist_refit": (780, 712, 68, 58.475698),
    "middle_q5_admission": (243, 216, 27, 31.886758),
    "crossvenue_middle_specialist": (312, 273, 39, 18.725328),
    "middle_agreement_ensemble": (415, 370, 45, 10.986521),
    "price_time_calibrated_middle_ensemble": (890, 815, 75, 78.969120),
    "extended_specialist_official": (95, 70, 25, 39.880590),
    "bridge_aware_specialist": (375, 314, 61, 37.528337),
    "official_vwap_admission": (811, 655, 156, 49.868085),
    "official_temporal_consensus": (78, 56, 22, 24.806127),
    "official_high_precision_loss_veto": (48, 33, 15, 18.391072),
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bucket_expr() -> pl.Expr:
    return (
        pl.when(pl.col("seconds_elapsed") < 90)
        .then(pl.lit("60_89"))
        .when(pl.col("seconds_elapsed") < 120)
        .then(pl.lit("90_119"))
        .when(pl.col("seconds_elapsed") < 150)
        .then(pl.lit("120_149"))
        .otherwise(pl.lit("150_180"))
    )


def _economic_columns(frame: pl.LazyFrame) -> pl.LazyFrame:
    return (
        frame.with_columns(
            _bucket_expr().alias("time_bucket"),
            pl.when(pl.col("probability") >= 0.5)
            .then(pl.lit("UP"))
            .otherwise(pl.lit("DOWN"))
            .alias("side"),
            pl.max_horizontal("probability", 1.0 - pl.col("probability")).alias(
                "selected_probability"
            ),
            (pl.col("probability") - 0.5).abs().mul(2.0).alias("confidence"),
            pl.when(pl.col("probability") >= 0.5)
            .then(pl.col("up_ask_vwap_5"))
            .otherwise(pl.col("down_ask_vwap_5"))
            .alias("share_cost"),
        )
        .with_columns(
            (pl.col("fee_rate") * pl.col("share_cost") * (1.0 - pl.col("share_cost"))).alias(
                "fee_per_share"
            ),
            ((pl.col("probability") >= 0.5) == (pl.col("label_up") == 1)).alias(
                "direction_correct"
            ),
        )
        .with_columns(
            (pl.col("selected_probability") - pl.col("share_cost") - pl.col("fee_per_share")).alias(
                "expected_edge"
            ),
            (
                pl.when(pl.col("direction_correct"))
                .then(1.0 - pl.col("share_cost") - pl.col("fee_per_share"))
                .otherwise(-pl.col("share_cost") - pl.col("fee_per_share"))
                * 5.0
            ).alias("net_pnl"),
        )
        .with_columns(
            (pl.col("net_pnl") - 0.05).alias("stress_net_pnl"),
            (pl.col("net_pnl") <= 0).cast(pl.Int8).alias("loss_label"),
        )
    )


def _history(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach strictly prior-market history; rows in a market share one state."""
    representative = (
        frame.sort(["champion", "window_start", "seconds_elapsed"])
        .group_by(["champion", "market_id"], maintain_order=True)
        .first()
        .sort(["champion", "window_start"])
    )
    records: list[dict[str, Any]] = []
    for champion, group in representative.partition_by("champion", as_dict=True).items():
        name = champion[0] if isinstance(champion, tuple) else champion
        recent: deque[dict[str, Any]] = deque(maxlen=25)
        loss_streak = 0
        for row in group.iter_rows(named=True):
            wins = [float(x["net_pnl"] > 0) for x in recent]
            briers = [
                (x["selected_probability"] - float(x["direction_correct"])) ** 2 for x in recent
            ]
            same_side = [x for x in recent if x["side"] == row["side"]]
            same_bucket = [x for x in recent if x["time_bucket"] == row["time_bucket"]]
            item = {
                "champion": name,
                "market_id": row["market_id"],
                "history_loss_streak": loss_streak,
            }
            for count in (5, 10, 25):
                sample = wins[-count:]
                item[f"history_win_rate_{count}"] = float(np.mean(sample)) if sample else None
            item.update(
                history_net_pnl_25=float(sum(x["net_pnl"] for x in recent)) if recent else None,
                history_stress_pnl_25=float(sum(x["stress_net_pnl"] for x in recent))
                if recent
                else None,
                history_brier_25=float(np.mean(briers)) if briers else None,
                history_expected_gap_25=(
                    float(np.mean([x["selected_probability"] for x in recent]) - np.mean(wins))
                    if recent
                    else None
                ),
                history_side_win_rate_25=(
                    float(np.mean([x["net_pnl"] > 0 for x in same_side])) if same_side else None
                ),
                history_bucket_win_rate_25=(
                    float(np.mean([x["net_pnl"] > 0 for x in same_bucket])) if same_bucket else None
                ),
            )
            records.append(item)
            loss_streak = loss_streak + 1 if row["net_pnl"] <= 0 else 0
            recent.append(row)
    return frame.join(pl.DataFrame(records), on=["champion", "market_id"], how="left")


def _paths(archive: Path) -> dict[str, Path]:
    results = archive / "training-results"
    return {
        "panel": archive
        / "data/btc-hybrid-payoff-admission-20260321-20260827/middle-panel.parquet",
        "full": results / "btc-5m-full-august-vwap-admission-20260321-20260901/20260901T163321Z",
        "extended": results
        / "btc-5m-extended-specialist-strategy-tournament-20260321-20260826/20260901T231526Z",
        "admission": results
        / "btc-5m-champion-admission-tournament-20260321-20260901/20260902T025934Z",
        "robust": results / "btc-5m-early-entry-robustness-20260321-20260901/20260902T191207Z",
    }


def build_construction_panel(archive: Path, maximum_rows: int) -> pl.DataFrame:
    paths = _paths(archive)
    base_columns = [
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "up_ask_vwap_5",
        "down_ask_vwap_5",
        "fee_rate",
        *MARKET_FEATURES,
        "pm_vwap5_overround",
        "pm_vwap50_overround",
        "pm_vwap200_overround",
        "pm_depth_imbalance",
        "pm_up_book_age_seconds",
        "pm_down_book_age_seconds",
    ]
    base = pl.scan_parquet(paths["panel"]).select(base_columns)
    prediction_paths = [
        (name, paths["full"] / f"ledgers/candidate-oof-predictions/{name}.parquet")
        for name in list(CHAMPIONS)[:5]
    ] + [
        (
            "extended_specialist_official",
            paths["extended"] / "ledgers/specialist-oof-predictions.parquet",
        ),
        ("bridge_aware_specialist", paths["admission"] / "ledgers/bridge-oof-predictions.parquet"),
    ]
    frames = []
    for champion, path in prediction_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        pred = (
            pl.scan_parquet(path)
            .filter(pl.col("eligible_signal"))
            .select("market_id", "observed_at", "probability", "fold")
        )
        frames.append(
            pred.join(base, on=["market_id", "observed_at"], how="inner").with_columns(
                pl.lit(champion).alias("champion")
            )
        )
    panel = (
        _economic_columns(pl.concat(frames, how="diagonal_relaxed"))
        .filter(
            pl.col("share_cost").is_between(0.0, 1.0, closed="none")
            & pl.col("fee_rate").is_not_null()
        )
        .collect(engine="streaming")
    )
    panel = panel.sort(["champion", "window_start", "seconds_elapsed"])
    if panel.height > maximum_rows:
        stride = math.ceil(panel.height / maximum_rows)
        panel = panel.with_row_index().filter(pl.col("index") % stride == 0).drop("index")
    return _history(panel)


def build_policy_panel(archive: Path) -> pl.DataFrame:
    paths = _paths(archive)
    common = tuple(
        dict.fromkeys(
            (
                "market_id",
                "window_start",
                "observed_at",
                "seconds_elapsed",
                "probability",
                "selected_probability",
                "share_cost",
                "fee_per_share",
                "expected_edge",
                "direction_correct",
                "net_pnl",
                "stress_net_pnl",
                "pm_vwap5_overround",
                "pm_vwap50_overround",
                "pm_vwap200_overround",
                "pm_depth_imbalance",
                "pm_up_book_age_seconds",
                "pm_down_book_age_seconds",
                "spot_l2_spread_bps",
                "kraken_l2_age_seconds",
                *MARKET_FEATURES,
            )
        )
    )
    frames: list[pl.LazyFrame] = []
    for champion, (source, mode, _, _) in CHAMPIONS.items():
        path = (
            paths[source]
            / "ledgers"
            / (
                "heldout-trades.parquet"
                if source in {"full", "extended"}
                else "sealed-trades.parquet"
            )
        )
        frame = pl.scan_parquet(path)
        schema = frame.collect_schema()
        if source in {"full", "extended"}:
            frame = frame.filter(
                (pl.col("candidate") == champion) & (pl.col("admission_mode") == mode)
            )
        else:
            frame = frame.filter(pl.col("strategy_candidate") == champion)
        if source == "robust":
            frame = frame.filter(pl.col("evaluation_bucket") != "combined_60_180")
        present = [column for column in common if column in schema]
        frame = frame.select(present).with_columns(pl.lit(champion).alias("champion"))
        for column in common:
            if column not in present:
                frame = frame.with_columns(pl.lit(None).cast(pl.Float64).alias(column))
        frames.append(frame.select("champion", *common))
    panel = (
        pl.concat(frames, how="diagonal_relaxed")
        .with_columns(
            _bucket_expr().alias("time_bucket"),
            pl.when(pl.col("probability") >= 0.5)
            .then(pl.lit("UP"))
            .otherwise(pl.lit("DOWN"))
            .alias("side"),
            (pl.col("probability") - 0.5).abs().mul(2.0).alias("confidence"),
            (pl.col("net_pnl") <= 0).cast(pl.Int8).alias("loss_label"),
        )
        .collect(engine="streaming")
    )
    return _history(panel.unique(subset=["champion", "market_id", "observed_at"], keep="first"))


class CalibratedRiskModel:
    def __init__(self, estimator: Any, features: tuple[str, ...]):
        self.estimator = estimator
        self.features = features
        self.calibrator = LogisticRegression(C=1.0, max_iter=500)

    def fit(self, fit: pl.DataFrame, calibration: pl.DataFrame) -> CalibratedRiskModel:
        self.estimator.fit(fit.select(self.features).to_numpy(), fit["loss_label"].to_numpy())
        raw = self.estimator.predict_proba(calibration.select(self.features).to_numpy())[:, 1]
        self.calibrator.fit(raw.reshape(-1, 1), calibration["loss_label"].to_numpy())
        return self

    def predict_proba(self, frame: pl.DataFrame) -> np.ndarray:
        raw = self.estimator.predict_proba(frame.select(self.features).to_numpy())[:, 1]
        return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def _models() -> dict[str, CalibratedRiskModel]:
    logistic = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.25, max_iter=800, class_weight="balanced")),
        ]
    )

    def boosted() -> Pipeline:
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_iter=160,
                        max_leaf_nodes=15,
                        min_samples_leaf=100,
                        l2_regularization=2.0,
                        random_state=20260909,
                    ),
                ),
            ]
        )

    return {
        "logistic_economics": CalibratedRiskModel(logistic, ECONOMIC_FEATURES),
        "boosted_economics": CalibratedRiskModel(boosted(), ECONOMIC_FEATURES),
        "boosted_market_context": CalibratedRiskModel(
            boosted(), ECONOMIC_FEATURES + MARKET_FEATURES
        ),
        "boosted_recent_history": CalibratedRiskModel(
            boosted(), ECONOMIC_FEATURES + MARKET_FEATURES + HISTORY_FEATURES
        ),
    }


def sequential_replay(frame: pl.DataFrame, allow: np.ndarray | pl.Series) -> pl.DataFrame:
    scored = frame.with_columns(pl.Series("risk_allowed", np.asarray(allow, dtype=bool)))
    return (
        scored.filter("risk_allowed")
        .sort(["champion", "window_start", "seconds_elapsed"])
        .group_by(["champion", "market_id"], maintain_order=True)
        .first()
    )


def trade_metrics(trades: pl.DataFrame, baseline_markets: int | None = None) -> dict[str, Any]:
    pnl = trades["net_pnl"].to_numpy() if trades.height else np.array([], dtype=float)
    stress = trades["stress_net_pnl"].to_numpy() if trades.height else np.array([], dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    equity = np.cumsum(pnl)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])
    drawdown = peaks[1:] - equity if len(equity) else np.array([])
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "trades": len(pnl),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else None,
        "net_pnl": float(pnl.sum()),
        "stress_net_pnl": float(stress.sum()),
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "expectancy_per_trade": float(pnl.mean()) if len(pnl) else None,
        "coverage_retained": float(len(pnl) / baseline_markets) if baseline_markets else None,
        "average_entry_seconds": float(trades["seconds_elapsed"].mean()) if trades.height else None,
        "average_cost": float(trades["share_cost"].mean()) if trades.height else None,
        "max_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "recovery_wins_per_loss": (
            float(abs(losses.mean()) / wins.mean()) if len(wins) and len(losses) else None
        ),
    }


def _predictive(y: np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    return {
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
    }


def _policy_evidence(
    frame: pl.DataFrame, probabilities: dict[str, np.ndarray]
) -> tuple[dict[str, Any], pl.DataFrame]:
    baseline = sequential_replay(frame, np.ones(frame.height, dtype=bool))
    baseline_metrics = trade_metrics(baseline, baseline.height)
    y = frame["loss_label"].to_numpy()
    evidence: dict[str, Any] = {"no_risk": {**baseline_metrics, "predictive": None}}
    scored = frame
    candidates: list[tuple[float, str, float, pl.DataFrame]] = []
    for name, probability in probabilities.items():
        scored = scored.with_columns(pl.Series(f"risk_probability__{name}", probability))
        best: tuple[float, float, pl.DataFrame] | None = None
        for threshold in np.quantile(probability, np.linspace(0.50, 0.95, 19)):
            trades = sequential_replay(frame, probability <= threshold)
            metrics = trade_metrics(trades, baseline.height)
            if metrics["coverage_retained"] < 0.50:
                continue
            score = metrics["stress_net_pnl"]
            if best is None or score > best[0]:
                best = (score, float(threshold), trades)
        if best is None:
            continue
        _, threshold, trades = best
        metrics = trade_metrics(trades, baseline.height)
        metrics.update(threshold=threshold, predictive=_predictive(y, probability))
        evidence[name] = metrics
        candidates.append((metrics["stress_net_pnl"], name, threshold, trades))
    candidates.sort(reverse=True, key=lambda item: item[0])
    _, selected_name, selected_threshold, selected_trades = candidates[0]
    selected_markets = set(
        zip(selected_trades["champion"], selected_trades["market_id"], strict=False)
    )
    blocked = baseline.with_columns(
        pl.Series(
            "blocked",
            [
                (champion, market) not in selected_markets
                for champion, market in zip(
                    baseline["champion"], baseline["market_id"], strict=False
                )
            ],
        )
    )
    evidence["selection"] = {"contestant": selected_name, "threshold": selected_threshold}
    evidence[selected_name]["avoided_loss_dollars"] = float(
        -blocked.filter(pl.col("blocked") & (pl.col("net_pnl") <= 0))["net_pnl"].sum()
    )
    evidence[selected_name]["missed_profit_dollars"] = float(
        blocked.filter(pl.col("blocked") & (pl.col("net_pnl") > 0))["net_pnl"].sum()
    )
    return evidence, scored


def _matched_controls(frame: pl.DataFrame, retained: float) -> dict[str, Any]:
    baseline = sequential_replay(frame, np.ones(frame.height, dtype=bool))
    output = {}
    for name, column in (("matched_confidence", "confidence"), ("matched_edge", "expected_edge")):
        threshold = float(frame[column].quantile(max(0.0, 1.0 - retained)))
        trades = sequential_replay(frame, frame[column].to_numpy() >= threshold)
        output[name] = {**trade_metrics(trades, baseline.height), "threshold": threshold}
    return output


def _slice_matrix(
    frame: pl.DataFrame, probability: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    scored = frame.with_columns(pl.Series("risk_allowed", probability <= threshold))
    rows = []
    for champion in CHAMPIONS:
        for bucket in BUCKETS:
            for side in ("UP", "DOWN"):
                part = scored.filter(
                    (pl.col("champion") == champion)
                    & (pl.col("time_bucket") == bucket)
                    & (pl.col("side") == side)
                )
                base = sequential_replay(part, np.ones(part.height, dtype=bool))
                risk = sequential_replay(part, part["risk_allowed"].to_numpy())
                base_keys = set(base["market_id"].to_list())
                risk_keys = set(risk["market_id"].to_list())
                blocked = base.filter(~pl.col("market_id").is_in(risk_keys))
                row = {
                    "champion": champion,
                    "time_bucket": bucket,
                    "side": side,
                    **trade_metrics(risk, len(base_keys)),
                }
                row.update(
                    baseline_trades=len(base_keys),
                    blocked_winners=int(blocked.filter(pl.col("net_pnl") > 0).height),
                    blocked_losses=int(blocked.filter(pl.col("net_pnl") <= 0).height),
                    avoided_loss_dollars=float(
                        -blocked.filter(pl.col("net_pnl") <= 0)["net_pnl"].sum()
                    ),
                    missed_profit_dollars=float(
                        blocked.filter(pl.col("net_pnl") > 0)["net_pnl"].sum()
                    ),
                    statistically_insufficient=len(base_keys) < 30,
                )
                rows.append(row)
    return rows


def _champion_comparison(
    frame: pl.DataFrame, probability: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    scored = frame.with_columns(pl.Series("risk_allowed", probability <= threshold))
    rows = []
    for champion in CHAMPIONS:
        part = scored.filter(pl.col("champion") == champion)
        baseline = sequential_replay(part, np.ones(part.height, dtype=bool))
        risk = sequential_replay(part, part["risk_allowed"].to_numpy())
        risk_keys = set(risk["market_id"].to_list())
        blocked = baseline.filter(~pl.col("market_id").is_in(risk_keys))
        row = {
            "champion": champion,
            "baseline": trade_metrics(baseline, baseline.height),
            "risk": trade_metrics(risk, baseline.height),
            "blocked_winners": blocked.filter(pl.col("net_pnl") > 0).height,
            "blocked_losses": blocked.filter(pl.col("net_pnl") <= 0).height,
            "avoided_loss_dollars": float(-blocked.filter(pl.col("net_pnl") <= 0)["net_pnl"].sum()),
            "missed_profit_dollars": float(blocked.filter(pl.col("net_pnl") > 0)["net_pnl"].sum()),
        }
        row["net_risk_value"] = row["risk"]["net_pnl"] - row["baseline"]["net_pnl"]
        rows.append(row)
    return rows


def _published_baseline_reproduction(frame: pl.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for champion, expected in PUBLISHED_BASELINES.items():
        part = frame.filter(
            (pl.col("champion") == champion) & (pl.col("time_bucket") == CHAMPIONS[champion][2])
        )
        actual = trade_metrics(
            sequential_replay(part, np.ones(part.height, dtype=bool)), expected[0]
        )
        passed = (actual["trades"], actual["wins"], actual["losses"]) == expected[:3] and abs(
            actual["net_pnl"] - expected[3]
        ) < 1e-5
        if not passed:
            raise RuntimeError(f"published baseline mismatch for {champion}: {actual}")
        rows.append({"champion": champion, "passed": passed, **actual})
    return rows


def _leave_one_champion_out(
    construction: pl.DataFrame,
    primary: pl.DataFrame,
    selected_name: str,
    threshold: float,
    calibration_start: datetime,
) -> list[dict[str, Any]]:
    rows = []
    construction_champions = set(construction["champion"].unique())
    for champion in CHAMPIONS:
        source = construction.filter(pl.col("champion") != champion)
        fit = source.filter(pl.col("window_start") < calibration_start)
        calibration = source.filter(pl.col("window_start") >= calibration_start)
        target = primary.filter(pl.col("champion") == champion)
        model = _models()[selected_name].fit(fit, calibration)
        probability = model.predict_proba(target)
        baseline = sequential_replay(target, np.ones(target.height, dtype=bool))
        risk = sequential_replay(target, probability <= threshold)
        rows.append(
            {
                "champion": champion,
                "held_out_from_construction": champion in construction_champions,
                "baseline": trade_metrics(baseline, baseline.height),
                "risk": trade_metrics(risk, baseline.height),
                "predictive": _predictive(target["loss_label"].to_numpy(), probability),
            }
        )
    return rows


def _report(metrics: dict[str, Any]) -> str:
    evidence = metrics["policy_selection"]
    lines = [
        "# BTC Candidate Loss-Risk Tournament",
        "",
        f"Run: `{metrics['run_id']}`",
        "",
        "## Contestants",
        "",
        "| Contestant | Threshold | Trades | W/L | PnL | Stress PnL | PF | Exp./trade | Coverage | Avg entry | Avg cost | Max DD | Recovery wins/loss | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in evidence.items():
        if name == "selection":
            continue
        predictive = value.get("predictive") or {}

        def f(key: str, digits: int = 3, row: dict[str, Any] = value) -> str:
            v = row.get(key)
            return "—" if v is None else f"{v:.{digits}f}"

        brier = "—" if predictive.get("brier") is None else f"{predictive['brier']:.4f}"
        lines.append(
            f"| {name} | {f('threshold')} | {value['trades']} | {value['wins']}/{value['losses']} | ${f('net_pnl', 2)} | ${f('stress_net_pnl', 2)} | {f('profit_factor')} | ${f('expectancy_per_trade')} | {f('coverage_retained', 3)} | {f('average_entry_seconds', 1)}s | ${f('average_cost')} | ${f('max_drawdown', 2)} | {f('recovery_wins_per_loss')} | {brier} |"
        )
    selected = evidence["selection"]
    lines += [
        "",
        "## Selection",
        "",
        f"Research winner: `{selected['contestant']}` at loss probability `{selected['threshold']:.6f}`.",
        "",
        "This is trained and evaluated research evidence only. It is not deployed or admitted for runtime use.",
        "",
        "## Champion comparison",
        "",
        "| Champion | Baseline PnL | Risk PnL | Risk W/L | Coverage | Avg entry | PF | Exp./trade | Max DD | Recovery wins/loss | Avoided losses | Missed profit | Net risk value |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["champion_comparison"]:
        risk = row["risk"]
        lines.append(
            f"| {row['champion']} | ${row['baseline']['net_pnl']:.2f} | ${risk['net_pnl']:.2f} | "
            f"{risk['wins']}/{risk['losses']} | {risk['coverage_retained']:.3f} | "
            f"{risk['average_entry_seconds']:.1f}s | {risk['profit_factor']:.3f} | "
            f"${risk['expectancy_per_trade']:.3f} | ${risk['max_drawdown']:.2f} | "
            f"{risk['recovery_wins_per_loss']:.3f} | ${row['avoided_loss_dollars']:.2f} | "
            f"${row['missed_profit_dollars']:.2f} | ${row['net_risk_value']:.2f} |"
        )
    lines += [
        "",
        "## Leave-one-champion-out diagnostic",
        "",
        "| Champion | Excluded from construction | Baseline PnL | Risk PnL | Coverage | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["leave_one_champion_out"]:
        lines.append(
            f"| {row['champion']} | "
            f"{'yes' if row['held_out_from_construction'] else 'unseen alias'} | "
            f"${row['baseline']['net_pnl']:.2f} | ${row['risk']['net_pnl']:.2f} | "
            f"{row['risk']['coverage_retained']:.3f} | {row['predictive']['brier']:.4f} |"
        )
    lines += [
        "",
        "## Published baseline reproduction",
        "",
        "All ten prior champion ledgers reproduced their published trade count, W/L, and PnL within $0.00001 before risk evaluation.",
        "",
        f"Qualification finding: **{metrics['qualification']['status']}**. "
        + ", ".join(
            f"{name}={'pass' if passed else 'fail'}"
            for name, passed in metrics["qualification"]["checks"].items()
        )
        + ".",
        "",
        "## Data limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in metrics["limitations"])
    return "\n".join(lines) + "\n"


def train_tournament(config_path: Path, resume_run: str | None = None) -> Path:
    raw = tomllib.loads(config_path.read_text())
    root = config_path.resolve().parents[1]
    archive = Path(raw["data"]["archive_root"])
    output_root = root / raw["output"]["directory"]
    run_id = resume_run or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    checkpoint = run_dir / "checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)
    construction_path = checkpoint / "construction-panel.parquet"
    policy_path = checkpoint / "policy-panel.parquet"
    if construction_path.exists():
        construction = pl.read_parquet(construction_path)
    else:
        construction = build_construction_panel(archive, int(raw["training"]["maximum_rows"]))
        construction.write_parquet(construction_path, compression="zstd", statistics=True)
    if policy_path.exists():
        policy = pl.read_parquet(policy_path)
    else:
        policy = build_policy_panel(archive)
        policy.write_parquet(policy_path, compression="zstd", statistics=True)
    calibration_start = datetime.fromisoformat(raw["splits"]["calibration_start"])
    policy_start = datetime.fromisoformat(raw["splits"]["policy_start"])
    fit = construction.filter(pl.col("window_start") < calibration_start)
    calibration = construction.filter(
        (pl.col("window_start") >= calibration_start) & (pl.col("window_start") < policy_start)
    )
    baseline_reproduction = _published_baseline_reproduction(policy)
    policy = policy.filter(pl.col("window_start") >= policy_start)
    if not set(fit["market_id"].unique()).isdisjoint(set(calibration["market_id"].unique())):
        raise RuntimeError("fit and calibration markets overlap")
    if not set(construction["market_id"].unique()).isdisjoint(set(policy["market_id"].unique())):
        raise RuntimeError("construction and policy markets overlap")
    models = _models()
    probabilities = {}
    for name, model in models.items():
        model.fit(fit, calibration)
        probabilities[name] = model.predict_proba(policy)
    preferred_buckets = {champion: values[2] for champion, values in CHAMPIONS.items()}
    primary_mask = (
        policy["time_bucket"]
        == policy["champion"].replace_strict(preferred_buckets, return_dtype=pl.String)
    ).to_numpy()
    primary = policy.filter(pl.Series(primary_mask))
    primary_probabilities = {name: values[primary_mask] for name, values in probabilities.items()}
    evidence, _ = _policy_evidence(primary, primary_probabilities)
    selected = evidence["selection"]
    selected_probability = probabilities[selected["contestant"]]
    selected_primary_probability = primary_probabilities[selected["contestant"]]
    retained = evidence[selected["contestant"]]["coverage_retained"]
    evidence.update(_matched_controls(primary, retained))
    slices = _slice_matrix(policy, selected_probability, selected["threshold"])
    comparison = _champion_comparison(primary, selected_primary_probability, selected["threshold"])
    loco_path = checkpoint / "leave-one-champion-out.json"
    if loco_path.exists():
        leave_one_out = json.loads(loco_path.read_text())
    else:
        leave_one_out = _leave_one_champion_out(
            construction,
            primary,
            selected["contestant"],
            selected["threshold"],
            calibration_start,
        )
        _write_json(loco_path, leave_one_out)
    scored = policy
    for name, probability in probabilities.items():
        scored = scored.with_columns(pl.Series(f"risk_probability__{name}", probability))
    ledgers = run_dir / "ledgers"
    ledgers.mkdir(exist_ok=True)
    scored.write_parquet(ledgers / "policy-candidates.parquet", compression="zstd", statistics=True)
    pl.DataFrame(slices).write_parquet(
        ledgers / "slice-matrix.parquet", compression="zstd", statistics=True
    )
    producing_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    artifact = run_dir / "tournament.joblib"
    serializable_models = {
        name: {
            "estimator": model.estimator,
            "calibrator": model.calibrator,
            "features": model.features,
        }
        for name, model in models.items()
    }
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "producing_commit": producing_commit,
            "models": serializable_models,
            "selected_contestant": selected["contestant"],
            "loss_probability_threshold": selected["threshold"],
            "features": models[selected["contestant"]].features,
            "deployment_status": "not_deployed",
        },
        artifact,
        compress=3,
    )
    artifact_sha = _sha256(artifact)
    (run_dir / "tournament.sha256").write_text(f"{artifact_sha}  tournament.joblib\n")
    source_paths = _paths(archive)
    selected_metrics = evidence[selected["contestant"]]
    baseline_metrics = evidence["no_risk"]
    confidence_control = evidence["matched_confidence"]
    edge_control = evidence["matched_edge"]
    qualification_checks = {
        "net_pnl_improved": selected_metrics["net_pnl"] > baseline_metrics["net_pnl"],
        "stress_pnl_improved": selected_metrics["stress_net_pnl"]
        > baseline_metrics["stress_net_pnl"],
        "max_drawdown_reduced": selected_metrics["max_drawdown"] < baseline_metrics["max_drawdown"],
        "avoided_losses_exceed_missed_profit": selected_metrics["avoided_loss_dollars"]
        > selected_metrics["missed_profit_dollars"],
        "coverage_floor": selected_metrics["coverage_retained"] >= 0.50,
        "beats_matched_controls": selected_metrics["net_pnl"]
        > max(confidence_control["net_pnl"], edge_control["net_pnl"]),
        "not_single_champion": sum(row["net_risk_value"] > 0 for row in comparison) > 1,
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "producing_commit": producing_commit,
        "artifact_sha256": artifact_sha,
        "qualification_status": "trained_evaluated_not_deployed",
        "deployment_status": "not_deployed",
        "source_identity": {
            name: {"path": str(path), "sha256": _sha256(path) if path.is_file() else None}
            for name, path in source_paths.items()
        },
        "splits": {
            "fit": [str(fit["window_start"].min()), str(fit["window_start"].max())],
            "calibration": [
                str(calibration["window_start"].min()),
                str(calibration["window_start"].max()),
            ],
            "policy": [str(policy["window_start"].min()), str(policy["window_start"].max())],
        },
        "row_counts": {
            "construction": construction.height,
            "fit": fit.height,
            "calibration": calibration.height,
            "policy": policy.height,
        },
        "policy_selection": evidence,
        "champion_comparison": comparison,
        "leave_one_champion_out": leave_one_out,
        "published_baseline_reproduction": baseline_reproduction,
        "qualification": {
            "status": (
                "research_qualified"
                if all(qualification_checks.values())
                else "research_not_qualified"
            ),
            "checks": qualification_checks,
        },
        "slice_matrix": slices,
        "integrity": {
            "market_disjoint": True,
            "oof_construction_predictions": True,
            "database_reads": False,
            "database_mutations": False,
            "new_tables": False,
            "new_schemas": False,
            "new_ingesters": False,
            "new_sources": False,
            "images_rebuilt": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "sklearn": sklearn.__version__,
        },
        "limitations": [
            "Causal OOF prediction ledgers begin April 1, so March 21-31 cannot contribute model-fitting rows.",
            "Construction uses all available chronological periods but a deterministic row stride caps memory; no date range is isolated.",
            "Exact base-admission ledgers are available only for August 14-25; August 26-31 lacks executable trade evidence.",
            "Three admission-layer champions share the extended specialist directional stream during construction; champion identity is not a model feature.",
            "The locked September replay is not opened because the frozen champion candidate/admission ledgers are not archived for that interval.",
            "Results are a pilot backtest and projected PnL assumes the archived five-share ask VWAP was fillable.",
        ],
    }
    _write_json(run_dir / "selection-freeze.json", selected)
    _write_json(run_dir / "metrics.json", metrics)
    (run_dir / "report.md").write_text(_report(metrics))
    _write_json(
        run_dir / "completion.json",
        {
            "completed": True,
            "artifact_sha256": artifact_sha,
            "metrics_sha256": _sha256(run_dir / "metrics.json"),
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-run")
    args = parser.parse_args()
    print(train_tournament(args.config, args.resume_run))


if __name__ == "__main__":
    main()
