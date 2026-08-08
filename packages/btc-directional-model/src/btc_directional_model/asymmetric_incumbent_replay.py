"""Frozen asymmetric-value incumbent scoring and policy replay."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import polars as pl

from .asymmetric_value_config import ValuePolicy
from .asymmetric_value_evaluation import policy_ledger, score_two_sided_value
from .asymmetric_value_training import (
    CORE_ORACLE_PRICE,
    CORE_PRICE,
    asymmetric_probability_frame,
    asymmetric_value_feature_sets,
)
from .core_extract import file_sha256
from .runtime_export import (
    ASYMMETRIC_VALUE_RUNTIME_MODEL_SCHEMA_VERSION,
    RUNTIME_MANIFEST_SCHEMA_VERSION,
    feature_schema_sha256,
    reached_leaf_value,
    stable_sigmoid,
)

FROZEN_ASYMMETRIC_INCUMBENT_KEY = (
    "btc-5m-asymmetric-core-oracle-paper-20260805-v1"
)
FROZEN_ASYMMETRIC_INCUMBENT_MODEL_SHA256 = (
    "2c91e894356f6fee7fe9514e24c39da6e11602ffcb7f961f64848bde72418db9"
)
DEFAULT_FROZEN_ASYMMETRIC_INCUMBENT_MODEL = (
    Path(__file__).resolve().parents[2]
    / "runtime-models"
    / FROZEN_ASYMMETRIC_INCUMBENT_KEY
    / "model.json"
)
FROZEN_ASYMMETRIC_INCUMBENT_POLICY = ValuePolicy(
    name="raw20_30_by55_edge_3c",
    selection_eligible=True,
    maximum_entry_second=55,
    minimum_share_price=0.20,
    maximum_share_price=0.30,
    maximum_cost_per_share=0.35,
    minimum_edge_per_share=0.03,
)
FROZEN_ASYMMETRIC_INCUMBENT_QUANTITY = 5.0
FROZEN_ASYMMETRIC_INCUMBENT_MAXIMUM_DEPTH_PARTICIPATION = 0.25
_PROBABILITY_CLIP = (1e-9, 1.0 - 1e-9)
_LOGIT_CLIP = (-40.0, 40.0)
_REPLAY_EXECUTION_COLUMNS = {
    "market_id",
    "window_start",
    "observed_at",
    "seconds_elapsed",
    "label_up",
    "fee_rate",
    "yes_best_ask",
    "yes_ask_vwap_5",
    "yes_ask_depth",
    "no_best_ask",
    "no_ask_vwap_5",
    "no_ask_depth",
    "yes_cost_per_share",
    "no_cost_per_share",
    "yes_execution_cost_per_share",
    "no_execution_cost_per_share",
}


@dataclass(frozen=True)
class FrozenAsymmetricRuntimeModel:
    """Validated immutable identity for one exported asymmetric runtime model."""

    path: Path
    payload: dict[str, Any]
    model_sha256: str
    manifest_sha256: str
    feature_contract: str

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.payload["features"]["names"])


@dataclass(frozen=True)
class AsymmetricIncumbentReplay:
    """Market-level output from the fixed lower-price incumbent policy."""

    model_key: str
    feature_contract: str
    selected_trades: pl.DataFrame
    eligible_resolved_markets: int
    trades_per_eligible_resolved_market: float
    audit_hashes: dict[str, str]


def load_frozen_asymmetric_incumbent(
    model_path: Path = DEFAULT_FROZEN_ASYMMETRIC_INCUMBENT_MODEL,
) -> FrozenAsymmetricRuntimeModel:
    """Load the named incumbent and fail if its tracked bytes or manifest changed."""

    path = model_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen asymmetric incumbent is missing: {path}")
    model_sha256 = file_sha256(path)
    if model_sha256 != FROZEN_ASYMMETRIC_INCUMBENT_MODEL_SHA256:
        raise RuntimeError("frozen asymmetric incumbent model SHA-256 changed")
    payload = _read_json_object(path, "frozen asymmetric incumbent model")
    feature_contract = validate_asymmetric_runtime_model(payload)
    if payload["model_key"] != FROZEN_ASYMMETRIC_INCUMBENT_KEY:
        raise RuntimeError("frozen asymmetric incumbent model key changed")

    manifest_path = path.with_name("manifest.json")
    manifest = _read_json_object(
        manifest_path,
        "frozen asymmetric incumbent manifest",
    )
    _validate_manifest(manifest, payload, model_sha256)
    return FrozenAsymmetricRuntimeModel(
        path=path,
        payload=payload,
        model_sha256=model_sha256,
        manifest_sha256=file_sha256(manifest_path),
        feature_contract=feature_contract,
    )


def validate_asymmetric_runtime_model(model: dict[str, Any]) -> str:
    """Validate the native asymmetric runtime contract and return its feature family."""

    if model.get("schema_version") != ASYMMETRIC_VALUE_RUNTIME_MODEL_SCHEMA_VERSION:
        raise ValueError("asymmetric runtime model schema is unsupported")
    model_key = model.get("model_key")
    if not isinstance(model_key, str) or not model_key:
        raise ValueError("asymmetric runtime model key is invalid")

    features = _require_object(model, "features")
    feature_names = features.get("names")
    medians = features.get("imputation_medians")
    if not isinstance(feature_names, list) or not all(
        isinstance(name, str) and name for name in feature_names
    ):
        raise ValueError("asymmetric runtime feature names are invalid")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("asymmetric runtime feature names are duplicated")
    feature_contracts = asymmetric_value_feature_sets()
    expected_by_count = {
        len(feature_contracts[CORE_PRICE]): CORE_PRICE,
        len(feature_contracts[CORE_ORACLE_PRICE]): CORE_ORACLE_PRICE,
    }
    feature_contract = expected_by_count.get(len(feature_names))
    if feature_contract is None or tuple(feature_names) != feature_contracts[feature_contract]:
        raise ValueError("asymmetric runtime feature order must match the 71/75 contract")
    if (
        not isinstance(medians, list)
        or len(medians) != len(feature_names)
        or not all(_is_finite_number(value) for value in medians)
    ):
        raise ValueError("asymmetric runtime imputation medians are invalid")
    schema_version = features.get("schema_version")
    if (
        not isinstance(schema_version, str)
        or not schema_version
        or features.get("schema_sha256")
        != feature_schema_sha256(schema_version, feature_names)
        or features.get("numeric_type") != "float64"
        or features.get("non_finite_policy") != "median_imputation"
    ):
        raise ValueError("asymmetric runtime feature schema identity is invalid")

    provenance = _require_object(model, "provenance")
    if provenance.get("candidate") != feature_contract:
        raise ValueError("asymmetric runtime candidate does not match its feature contract")
    for field in ("source_benchmark_sha256", "source_training_model_sha256"):
        if not _is_sha256(provenance.get(field)):
            raise ValueError(f"asymmetric runtime provenance hash is invalid: {field}")

    deployment = _require_object(model, "deployment")
    if deployment != {
        "scope": "paper_only",
        "production_qualified": False,
        "live_capital_allowed": False,
    }:
        raise ValueError("asymmetric runtime incumbent must remain paper-only")

    estimator = _require_object(model, "estimator")
    if (
        estimator.get("type") != "histogram_gradient_boosting_binary_classifier"
        or estimator.get("class_order") != [0, 1]
        or estimator.get("output") != "raw_logit"
        or estimator.get("tree_values_include_learning_rate") is not True
        or estimator.get("split_comparison") != "less_than_or_equal"
        or not _is_finite_number(estimator.get("baseline_logit"))
    ):
        raise ValueError("asymmetric runtime histogram estimator contract is invalid")
    trees = estimator.get("trees")
    if not isinstance(trees, list) or not trees:
        raise ValueError("asymmetric runtime histogram estimator has no trees")
    for index, tree in enumerate(trees):
        _validate_tree(tree, len(feature_names), index)

    decision = _require_object(model, "decision")
    if (
        decision.get("probability_up_threshold") != 0.5
        or decision.get("below_confidence_action") != "no_trade"
        or decision.get("up_action") != "up"
        or decision.get("down_action") != "down"
        or "confidence_threshold" in decision
    ):
        raise ValueError("asymmetric runtime decision contract is invalid")

    prediction_policy = _require_object(model, "prediction_policy")
    if prediction_policy != {
        "type": "scheduled",
        "minimum_seconds_after_open": 1,
        "maximum_seconds_after_open": 240,
        "early_end_second": 59,
        "early_cadence_seconds": 1,
        "cadence_seconds": 5,
    }:
        raise ValueError("asymmetric runtime prediction schedule changed")

    calibration = _require_object(model, "asymmetric_value_calibration")
    _validate_asymmetric_calibration(calibration, prediction_policy)
    for unsupported in ("calibration", "time_bands", "target"):
        if unsupported in model:
            raise ValueError("asymmetric runtime model contains directional calibration")
    return feature_contract


def score_asymmetric_runtime_row(
    model: FrozenAsymmetricRuntimeModel,
    feature_values: Sequence[float | int | None],
    *,
    seconds_elapsed: int,
    yes_ask_vwap: float,
    no_ask_vwap: float,
) -> dict[str, Any]:
    """Score one row in the same order as native asymmetric runtime inference."""

    payload = model.payload
    features = payload["features"]
    if len(feature_values) != len(features["names"]):
        raise ValueError("asymmetric runtime feature count does not match model")
    if isinstance(seconds_elapsed, bool) or not isinstance(seconds_elapsed, int):
        raise TypeError("asymmetric runtime seconds_elapsed must be an integer")
    if not _prediction_policy_accepts(payload["prediction_policy"], seconds_elapsed):
        raise ValueError("seconds_elapsed is outside the asymmetric prediction schedule")
    yes_price = _valid_price(yes_ask_vwap, "YES")
    no_price = _valid_price(no_ask_vwap, "NO")

    values: list[float] = []
    for value, median in zip(
        feature_values,
        features["imputation_medians"],
        strict=True,
    ):
        numeric = (
            float(value)
            if value is not None and not isinstance(value, bool)
            else math.nan
        )
        values.append(numeric if math.isfinite(numeric) else float(median))

    estimator = payload["estimator"]
    raw_logit = float(estimator["baseline_logit"])
    for tree in estimator["trees"]:
        raw_logit += reached_leaf_value(tree["nodes"], values)
    if not math.isfinite(raw_logit):
        raise RuntimeError("asymmetric runtime produced a non-finite raw logit")

    calibration = payload["asymmetric_value_calibration"]
    time_band = _time_band(calibration["time_bands"], seconds_elapsed)
    parent_eta = _clip(
        raw_logit * float(time_band["slope"]) + float(time_band["intercept"]),
        *_LOGIT_CLIP,
    )
    parent_yes = stable_sigmoid(parent_eta)
    parent_yes = _clip(parent_yes, *_PROBABILITY_CLIP)
    parent_no = 1.0 - parent_yes
    yes_logit = math.log(parent_yes / (1.0 - parent_yes))
    no_logit = math.log(parent_no / (1.0 - parent_no))
    cells = calibration["side_price_cells"]
    yes_cell = _price_cell(cells, "yes", seconds_elapsed, yes_price)
    no_cell = _price_cell(cells, "no", seconds_elapsed, no_price)
    yes_eta = yes_logit * float(yes_cell["slope"]) + float(yes_cell["intercept"])
    no_eta = no_logit * float(no_cell["slope"]) + float(no_cell["intercept"])
    coherent_eta = _clip(0.5 * (yes_eta - no_eta), *_LOGIT_CLIP)
    probability_up = stable_sigmoid(coherent_eta)
    if not math.isfinite(probability_up):
        raise RuntimeError("asymmetric runtime produced a non-finite coherent probability")
    return {
        "raw_logit": raw_logit,
        "probability_up": probability_up,
        "confidence": max(probability_up, 1.0 - probability_up),
        "action": "up" if probability_up >= 0.5 else "down",
        "accepted": True,
    }


def replay_frozen_asymmetric_incumbent(
    policy_frame: pl.DataFrame,
    *,
    model_path: Path = DEFAULT_FROZEN_ASYMMETRIC_INCUMBENT_MODEL,
) -> AsymmetricIncumbentReplay:
    """Replay the frozen 20-30c-by-55 policy on strict policy-window rows.

    The denominator contains resolved markets with at least one scorable observation
    no later than second 55. Price, depth, cost, and modeled-edge filters only affect
    the numerator, so the reported rate exposes the incumbent's actual selectivity.
    """

    model = load_frozen_asymmetric_incumbent(model_path)
    required = {*model.feature_names, *_REPLAY_EXECUTION_COLUMNS}
    missing = sorted(required - set(policy_frame.columns))
    if missing:
        raise ValueError("incumbent replay frame is missing columns: " + ", ".join(missing))
    if policy_frame.is_empty():
        raise ValueError("incumbent replay frame is empty")
    if not policy_frame.schema["seconds_elapsed"].is_integer():
        raise TypeError("incumbent replay seconds_elapsed must be an integer column")
    if not policy_frame.schema["label_up"].is_integer():
        raise TypeError("incumbent replay label_up must be an integer column")

    early = policy_frame.filter(
        pl.col("seconds_elapsed").is_between(
            model.payload["prediction_policy"]["minimum_seconds_after_open"],
            FROZEN_ASYMMETRIC_INCUMBENT_POLICY.maximum_entry_second,
            closed="both",
        )
    )
    if early.is_empty():
        raise ValueError("incumbent replay has no rows inside the fixed entry window")
    invalid_identity = early.filter(
        pl.col("market_id").is_null()
        | pl.col("window_start").is_null()
        | pl.col("observed_at").is_null()
    )
    if invalid_identity.height:
        raise ValueError("incumbent replay market identity is incomplete")
    invalid_labels = early.filter(
        pl.col("label_up").is_null() | ~pl.col("label_up").is_in([0, 1])
    )
    if invalid_labels.height:
        raise ValueError("incumbent replay requires resolved binary outcomes")
    inconsistent_markets = (
        early.group_by("market_id")
        .agg(
            pl.col("window_start").n_unique().alias("window_starts"),
            pl.col("label_up").n_unique().alias("labels"),
        )
        .filter((pl.col("window_starts") != 1) | (pl.col("labels") != 1))
    )
    if inconsistent_markets.height:
        raise ValueError("incumbent replay market identity or outcome is inconsistent")
    duplicate_points = (
        early.group_by("market_id", "window_start", "seconds_elapsed")
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate_points.height:
        raise ValueError("incumbent replay contains duplicate market decision points")

    matrix = early.select(*model.feature_names).to_numpy()
    probabilities = [
        score_asymmetric_runtime_row(
            model,
            row.tolist(),
            seconds_elapsed=int(second),
            yes_ask_vwap=yes_price,
            no_ask_vwap=no_price,
        )["probability_up"]
        for row, second, yes_price, no_price in zip(
            matrix,
            early["seconds_elapsed"],
            early["yes_ask_vwap_5"],
            early["no_ask_vwap_5"],
            strict=True,
        )
    ]
    predictions = asymmetric_probability_frame(
        early,
        probabilities,
        model=model.payload["model_key"],
    )
    selected = policy_ledger(
        score_two_sided_value(predictions),
        FROZEN_ASYMMETRIC_INCUMBENT_POLICY,
        quantity=FROZEN_ASYMMETRIC_INCUMBENT_QUANTITY,
        maximum_depth_participation=(
            FROZEN_ASYMMETRIC_INCUMBENT_MAXIMUM_DEPTH_PARTICIPATION
        ),
    )
    eligible_resolved_markets = early["market_id"].n_unique()
    if selected.height > eligible_resolved_markets:
        raise RuntimeError("incumbent replay selected more than one trade per market")
    trade_rate = selected.height / eligible_resolved_markets
    provenance = model.payload["provenance"]
    return AsymmetricIncumbentReplay(
        model_key=model.payload["model_key"],
        feature_contract=model.feature_contract,
        selected_trades=selected,
        eligible_resolved_markets=eligible_resolved_markets,
        trades_per_eligible_resolved_market=trade_rate,
        audit_hashes={
            "model_artifact_sha256": model.model_sha256,
            "runtime_manifest_sha256": model.manifest_sha256,
            "feature_schema_sha256": model.payload["features"]["schema_sha256"],
            "source_benchmark_sha256": provenance["source_benchmark_sha256"],
            "source_training_model_sha256": provenance[
                "source_training_model_sha256"
            ],
            "policy_contract_sha256": _policy_sha256(),
        },
    )


def _validate_manifest(
    manifest: dict[str, Any],
    model: dict[str, Any],
    model_sha256: str,
) -> None:
    features = model["features"]
    provenance = model["provenance"]
    expected = {
        "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "model_key": model["model_key"],
        "model_file": "model.json",
        "model_sha256": model_sha256,
        "feature_schema_version": features["schema_version"],
        "feature_schema_sha256": features["schema_sha256"],
        "source_freeze_manifest_sha256": provenance["source_benchmark_sha256"],
        "source_training_model_sha256": provenance["source_training_model_sha256"],
        "deployment_scope": "paper_only",
        "production_qualified": False,
        "live_capital_allowed": False,
    }
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatched:
        raise RuntimeError(
            "frozen asymmetric incumbent manifest mismatch: " + ", ".join(mismatched)
        )


def _validate_asymmetric_calibration(
    calibration: dict[str, Any],
    prediction_policy: dict[str, Any],
) -> None:
    bands = calibration.get("time_bands")
    cells = calibration.get("side_price_cells")
    if not isinstance(bands, list) or not bands:
        raise ValueError("asymmetric runtime time calibration is empty")
    if not isinstance(cells, list) or not cells:
        raise ValueError("asymmetric runtime side/price calibration is empty")

    expected_start = prediction_policy["minimum_seconds_after_open"]
    for band in bands:
        if not isinstance(band, dict):
            raise TypeError("asymmetric runtime time calibration band is invalid")
        start = band.get("start_seconds")
        end = band.get("end_seconds_exclusive")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start != expected_start
            or start >= end
            or not _is_finite_number(band.get("slope"))
            or float(band["slope"]) <= 0.0
            or not _is_finite_number(band.get("intercept"))
        ):
            raise ValueError("asymmetric runtime time calibration band is invalid")
        expected_start = end
    if expected_start != prediction_policy["maximum_seconds_after_open"] + 1:
        raise ValueError("asymmetric runtime time calibration does not cover schedule")

    for cell in cells:
        if not isinstance(cell, dict):
            raise TypeError("asymmetric runtime price calibration cell is invalid")
        minimum = cell.get("minimum_price")
        maximum = cell.get("maximum_price")
        if (
            cell.get("side") not in {"yes", "no"}
            or isinstance(cell.get("start_seconds"), bool)
            or not isinstance(cell.get("start_seconds"), int)
            or isinstance(cell.get("end_seconds_exclusive"), bool)
            or not isinstance(cell.get("end_seconds_exclusive"), int)
            or not _is_finite_number(minimum)
            or not _is_finite_number(maximum)
            or not 0.0 <= float(minimum) < float(maximum) <= 1.0
            or not _is_finite_number(cell.get("slope"))
            or float(cell["slope"]) <= 0.0
            or not _is_finite_number(cell.get("intercept"))
        ):
            raise ValueError("asymmetric runtime price calibration cell is invalid")

    for band in bands:
        for side in ("yes", "no"):
            matching = sorted(
                (
                    cell
                    for cell in cells
                    if cell["side"] == side
                    and cell["start_seconds"] == band["start_seconds"]
                    and cell["end_seconds_exclusive"]
                    == band["end_seconds_exclusive"]
                ),
                key=lambda cell: cell["minimum_price"],
            )
            if (
                len(matching) != 10
                or matching[0]["minimum_price"] != 0.0
                or matching[-1]["maximum_price"] != 1.0
                or any(
                    left["maximum_price"] != right["minimum_price"]
                    for left, right in pairwise(matching)
                )
            ):
                raise ValueError(
                    "asymmetric runtime price cells do not cover each side/time band"
                )
    if len(cells) != len(bands) * 20:
        raise ValueError("asymmetric runtime price calibration has unexpected cells")


def _validate_tree(tree: Any, feature_count: int, tree_index: int) -> None:
    if not isinstance(tree, dict) or not isinstance(tree.get("nodes"), list):
        raise TypeError(f"asymmetric runtime tree {tree_index} is invalid")
    nodes = tree["nodes"]
    if not nodes:
        raise ValueError(f"asymmetric runtime tree {tree_index} is empty")
    parents = [0] * len(nodes)
    for node_index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise TypeError(f"asymmetric runtime tree {tree_index} node is invalid")
        if node.get("kind") == "leaf":
            if not _is_finite_number(node.get("value")):
                raise ValueError(f"asymmetric runtime tree {tree_index} leaf is invalid")
            continue
        if node.get("kind") != "split":
            raise ValueError(f"asymmetric runtime tree {tree_index} node kind is invalid")
        feature_index = node.get("feature_index")
        left = node.get("left")
        right = node.get("right")
        if (
            isinstance(feature_index, bool)
            or not isinstance(feature_index, int)
            or not 0 <= feature_index < feature_count
            or isinstance(left, bool)
            or not isinstance(left, int)
            or isinstance(right, bool)
            or not isinstance(right, int)
            or not 0 <= left < len(nodes)
            or not 0 <= right < len(nodes)
            or left == right
            or left == node_index
            or right == node_index
            or not _is_finite_number(node.get("threshold"))
            or not isinstance(node.get("missing_go_to_left"), bool)
        ):
            raise ValueError(f"asymmetric runtime tree {tree_index} split is invalid")
        parents[left] += 1
        parents[right] += 1
    if parents[0] != 0 or any(count != 1 for count in parents[1:]):
        raise ValueError(f"asymmetric runtime tree {tree_index} is not rooted")

    visited: set[int] = set()
    stack = [0]
    while stack:
        index = stack.pop()
        if index in visited:
            raise ValueError(f"asymmetric runtime tree {tree_index} contains a cycle")
        visited.add(index)
        node = nodes[index]
        if node["kind"] == "split":
            stack.extend((node["right"], node["left"]))
    if len(visited) != len(nodes):
        raise ValueError(f"asymmetric runtime tree {tree_index} has unreachable nodes")


def _prediction_policy_accepts(policy: dict[str, Any], seconds_elapsed: int) -> bool:
    minimum = int(policy["minimum_seconds_after_open"])
    maximum = int(policy["maximum_seconds_after_open"])
    if not minimum <= seconds_elapsed <= maximum:
        return False
    early_end = int(policy["early_end_second"])
    if seconds_elapsed <= early_end:
        start = minimum
        cadence = int(policy["early_cadence_seconds"])
    else:
        start = early_end + 1
        cadence = int(policy["cadence_seconds"])
    return (seconds_elapsed - start) % cadence == 0


def _time_band(bands: list[dict[str, Any]], second: int) -> dict[str, Any]:
    for band in bands:
        if band["start_seconds"] <= second < band["end_seconds_exclusive"]:
            return band
    raise ValueError("asymmetric runtime has no calibration time band")


def _price_cell(
    cells: list[dict[str, Any]],
    side: str,
    second: int,
    price: float,
) -> dict[str, Any]:
    for cell in cells:
        if (
            cell["side"] == side
            and cell["start_seconds"] <= second < cell["end_seconds_exclusive"]
            and price >= cell["minimum_price"]
            and (
                price < cell["maximum_price"]
                or (cell["maximum_price"] == 1.0 and price <= 1.0)
            )
        ):
            return cell
    raise ValueError(f"asymmetric runtime has no {side.upper()} price calibration cell")


def _valid_price(value: Any, side: str) -> float:
    if isinstance(value, bool) or not _is_finite_number(value):
        raise ValueError(f"asymmetric runtime {side} price is invalid")
    price = float(value)
    if not 0.0 <= price <= 1.0:
        raise ValueError(f"asymmetric runtime {side} price is invalid")
    return price


def _policy_sha256() -> str:
    payload = {
        "policy": asdict(FROZEN_ASYMMETRIC_INCUMBENT_POLICY),
        "quantity": FROZEN_ASYMMETRIC_INCUMBENT_QUANTITY,
        "maximum_depth_participation": (
            FROZEN_ASYMMETRIC_INCUMBENT_MAXIMUM_DEPTH_PARTICIPATION
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _require_object(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise TypeError(f"asymmetric runtime field must be an object: {name}")
    return value


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _clip(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)
