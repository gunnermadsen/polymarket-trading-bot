from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .core_config import parse_utc_day

LEGACY_OFFLINE_BENCHMARK_MODE = "legacy_offline_challengers"
CORE_ONLY_REUSE_DIAGNOSTICS_MODE = "core_only_reuse_diagnostics"
STRICT_BOOK_CHRONOLOGICAL_MODE = "strict_book_chronological"


@dataclass(frozen=True)
class BenchmarkIdentityConfig:
    core_config: Path
    control_candidate: str
    early_candidate: str
    strict_book_candidate: str
    evaluation_is_independent: bool
    evaluation_note: str
    fixed_evaluation_seconds: tuple[int, ...]
    quantity: float
    mode: str
    candidate_names: tuple[str, ...]


@dataclass(frozen=True)
class PriorDiagnosticsConfig:
    record: Path
    sha256: str
    run_id: str


@dataclass(frozen=True)
class BenchmarkExecutionConfig:
    range_start: datetime
    range_end: datetime
    sample_interval_seconds: int
    min_seconds_after_open: int
    max_seconds_after_open: int
    stale_after_seconds: int
    output_dir: Path


@dataclass(frozen=True)
class BenchmarkBookSplitConfig:
    fit_start: datetime
    fit_end: datetime
    calibration_start: datetime
    calibration_end: datetime
    policy_start: datetime
    policy_end: datetime


@dataclass(frozen=True)
class BenchmarkBookModelConfig:
    confidence_min: float
    confidence_max: float
    confidence_step: float


@dataclass(frozen=True)
class BenchmarkBookEvaluationConfig:
    range_start: datetime
    range_end: datetime
    output_dir: Path


@dataclass(frozen=True)
class BenchmarkGateConfig:
    minimum_accuracy: float
    minimum_balanced_accuracy: float
    minimum_direction_recall: float
    minimum_wilson_lower_95: float
    maximum_expected_calibration_error: float
    minimum_coverage: float
    minimum_coverage_uplift: float
    maximum_accuracy_regression: float
    maximum_balanced_accuracy_regression: float
    maximum_direction_recall_regression: float
    maximum_median_entry_seconds_regression: float
    minimum_executable_markets: int
    minimum_mean_direct_edge_per_share: float
    minimum_realized_net_per_share: float
    minimum_common_time_markets: int
    maximum_native_p99_milliseconds: float
    maximum_runtime_model_bytes: int


@dataclass(frozen=True)
class BenchmarkComputeConfig:
    max_parallel_candidates: int
    threads_per_fit: int
    polars_threads: int


@dataclass(frozen=True)
class BenchmarkPathConfig:
    preopen_features: Path
    runs: Path
    artifacts: Path


@dataclass(frozen=True)
class EntryBenchmarkConfig:
    source_path: Path
    package_root: Path
    benchmark: BenchmarkIdentityConfig
    execution: BenchmarkExecutionConfig
    book_split: BenchmarkBookSplitConfig
    book_model: BenchmarkBookModelConfig
    gates: BenchmarkGateConfig
    compute: BenchmarkComputeConfig
    paths: BenchmarkPathConfig
    prior_diagnostics: PriorDiagnosticsConfig | None
    book_evaluation: BenchmarkBookEvaluationConfig | None


def load_entry_benchmark_config(path: Path) -> EntryBenchmarkConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)

    benchmark_raw = raw["benchmark"]
    execution_raw = raw["execution"]
    split_raw = raw["book_split"]
    book_model_raw = raw["book_model"]
    gates_raw = raw["advancement_gates"]
    compute_raw = raw["compute"]
    paths_raw = raw["paths"]
    prior_raw = raw.get("prior_diagnostics")
    book_evaluation_raw = raw.get("book_evaluation")
    control_candidate = str(benchmark_raw["control_candidate"])
    early_candidate = str(benchmark_raw["early_candidate"])
    config = EntryBenchmarkConfig(
        source_path=source_path,
        package_root=package_root,
        benchmark=BenchmarkIdentityConfig(
            core_config=package_root / str(benchmark_raw["core_config"]),
            control_candidate=control_candidate,
            early_candidate=early_candidate,
            strict_book_candidate=str(benchmark_raw["strict_book_candidate"]),
            evaluation_is_independent=bool(
                benchmark_raw["evaluation_is_independent"]
            ),
            evaluation_note=str(benchmark_raw["evaluation_note"]).strip(),
            fixed_evaluation_seconds=tuple(
                int(value) for value in benchmark_raw["fixed_evaluation_seconds"]
            ),
            quantity=float(benchmark_raw["quantity"]),
            mode=str(
                benchmark_raw.get(
                    "mode",
                    LEGACY_OFFLINE_BENCHMARK_MODE,
                )
            ),
            candidate_names=tuple(
                str(value)
                for value in benchmark_raw.get(
                    "candidate_names",
                    (control_candidate, early_candidate),
                )
            ),
        ),
        execution=BenchmarkExecutionConfig(
            range_start=parse_utc_day(execution_raw["range_start"]),
            range_end=parse_utc_day(execution_raw["range_end"]),
            sample_interval_seconds=int(execution_raw["sample_interval_seconds"]),
            min_seconds_after_open=int(execution_raw["min_seconds_after_open"]),
            max_seconds_after_open=int(execution_raw["max_seconds_after_open"]),
            stale_after_seconds=int(execution_raw["stale_after_seconds"]),
            output_dir=package_root / str(execution_raw["output_dir"]),
        ),
        book_split=BenchmarkBookSplitConfig(
            fit_start=parse_utc_day(split_raw["fit_start"]),
            fit_end=parse_utc_day(split_raw["fit_end"]),
            calibration_start=parse_utc_day(split_raw["calibration_start"]),
            calibration_end=parse_utc_day(split_raw["calibration_end"]),
            policy_start=parse_utc_day(split_raw["policy_start"]),
            policy_end=parse_utc_day(split_raw["policy_end"]),
        ),
        book_model=BenchmarkBookModelConfig(
            confidence_min=float(book_model_raw["confidence_min"]),
            confidence_max=float(book_model_raw["confidence_max"]),
            confidence_step=float(book_model_raw["confidence_step"]),
        ),
        gates=BenchmarkGateConfig(
            minimum_accuracy=float(gates_raw["minimum_accuracy"]),
            minimum_balanced_accuracy=float(
                gates_raw["minimum_balanced_accuracy"]
            ),
            minimum_direction_recall=float(
                gates_raw["minimum_direction_recall"]
            ),
            minimum_wilson_lower_95=float(
                gates_raw["minimum_wilson_lower_95"]
            ),
            maximum_expected_calibration_error=float(
                gates_raw["maximum_expected_calibration_error"]
            ),
            minimum_coverage=float(gates_raw["minimum_coverage"]),
            minimum_coverage_uplift=float(
                gates_raw["minimum_coverage_uplift"]
            ),
            maximum_accuracy_regression=float(
                gates_raw["maximum_accuracy_regression"]
            ),
            maximum_balanced_accuracy_regression=float(
                gates_raw["maximum_balanced_accuracy_regression"]
            ),
            maximum_direction_recall_regression=float(
                gates_raw["maximum_direction_recall_regression"]
            ),
            maximum_median_entry_seconds_regression=float(
                gates_raw["maximum_median_entry_seconds_regression"]
            ),
            minimum_executable_markets=int(
                gates_raw["minimum_executable_markets"]
            ),
            minimum_mean_direct_edge_per_share=float(
                gates_raw["minimum_mean_direct_edge_per_share"]
            ),
            minimum_realized_net_per_share=float(
                gates_raw["minimum_realized_net_per_share"]
            ),
            minimum_common_time_markets=int(
                gates_raw["minimum_common_time_markets"]
            ),
            maximum_native_p99_milliseconds=float(
                gates_raw["maximum_native_p99_milliseconds"]
            ),
            maximum_runtime_model_bytes=int(
                gates_raw["maximum_runtime_model_bytes"]
            ),
        ),
        compute=BenchmarkComputeConfig(
            max_parallel_candidates=int(
                compute_raw["max_parallel_candidates"]
            ),
            threads_per_fit=int(compute_raw["threads_per_fit"]),
            polars_threads=int(compute_raw["polars_threads"]),
        ),
        paths=BenchmarkPathConfig(
            preopen_features=package_root
            / str(paths_raw["preopen_features"]),
            runs=package_root / str(paths_raw["runs"]),
            artifacts=package_root / str(paths_raw["artifacts"]),
        ),
        prior_diagnostics=(
            PriorDiagnosticsConfig(
                record=package_root / str(prior_raw["record"]),
                sha256=str(prior_raw["sha256"]).lower(),
                run_id=str(prior_raw["run_id"]),
            )
            if prior_raw is not None
            else None
        ),
        book_evaluation=(
            BenchmarkBookEvaluationConfig(
                range_start=parse_utc_day(book_evaluation_raw["range_start"]),
                range_end=parse_utc_day(book_evaluation_raw["range_end"]),
                output_dir=package_root
                / str(book_evaluation_raw["output_dir"]),
            )
            if book_evaluation_raw is not None
            else None
        ),
    )
    validate_entry_benchmark_config(config)
    return config


def validate_entry_benchmark_config(config: EntryBenchmarkConfig) -> None:
    identity = config.benchmark
    execution = config.execution
    split = config.book_split
    if not identity.core_config.is_file():
        raise ValueError(f"core config is missing: {identity.core_config}")
    if identity.mode not in {
        LEGACY_OFFLINE_BENCHMARK_MODE,
        CORE_ONLY_REUSE_DIAGNOSTICS_MODE,
        STRICT_BOOK_CHRONOLOGICAL_MODE,
    }:
        raise ValueError(f"unsupported benchmark mode: {identity.mode}")
    candidates = (
        identity.control_candidate,
        identity.early_candidate,
        identity.strict_book_candidate,
    )
    if any(not candidate.strip() for candidate in candidates):
        raise ValueError("benchmark candidate names must be non-empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("benchmark candidate names must be unique")
    if (
        not identity.candidate_names
        or any(not candidate.strip() for candidate in identity.candidate_names)
        or len(set(identity.candidate_names)) != len(identity.candidate_names)
    ):
        raise ValueError("active candidate_names must be non-empty and unique")
    if identity.control_candidate not in identity.candidate_names:
        raise ValueError("control_candidate must be in active candidate_names")
    if (
        identity.mode == LEGACY_OFFLINE_BENCHMARK_MODE
        and identity.candidate_names
        != (identity.control_candidate, identity.early_candidate)
    ):
        raise ValueError(
            "legacy benchmark candidate_names must remain control plus early"
        )
    if (
        identity.mode == STRICT_BOOK_CHRONOLOGICAL_MODE
        and identity.candidate_names
        != (identity.control_candidate, identity.strict_book_candidate)
    ):
        raise ValueError(
            "strict-book chronological benchmark candidates must be "
            "the BTC-only control plus strict-book challenger"
        )
    if (
        identity.mode == STRICT_BOOK_CHRONOLOGICAL_MODE
        and config.book_evaluation is None
    ):
        raise ValueError(
            "strict-book chronological benchmark requires book_evaluation"
        )
    if (
        identity.mode != STRICT_BOOK_CHRONOLOGICAL_MODE
        and config.book_evaluation is not None
    ):
        raise ValueError(
            "book_evaluation is only valid for strict-book chronological mode"
        )
    if (
        identity.mode == CORE_ONLY_REUSE_DIAGNOSTICS_MODE
        and config.prior_diagnostics is None
    ):
        raise ValueError("core-only benchmark requires pinned prior diagnostics")
    if config.prior_diagnostics is not None:
        diagnostics = config.prior_diagnostics
        if (
            len(diagnostics.sha256) != 64
            or any(character not in "0123456789abcdef" for character in diagnostics.sha256)
        ):
            raise ValueError("prior diagnostics sha256 must be 64 lowercase hex digits")
        if not diagnostics.run_id.strip():
            raise ValueError("prior diagnostics run_id must be non-empty")
    if not identity.evaluation_note:
        raise ValueError("benchmark evaluation note is required")
    if identity.evaluation_is_independent:
        raise ValueError(
            "this consumed historical range must remain marked non-independent"
        )
    if identity.quantity <= 0:
        raise ValueError("benchmark quantity must be positive")
    if (
        not identity.fixed_evaluation_seconds
        or tuple(sorted(set(identity.fixed_evaluation_seconds)))
        != identity.fixed_evaluation_seconds
    ):
        raise ValueError("fixed evaluation seconds must be unique and sorted")
    if execution.range_start >= execution.range_end:
        raise ValueError("execution evidence range must be positive")
    if execution.sample_interval_seconds <= 0:
        raise ValueError("execution sample interval must be positive")
    if not 0 <= execution.min_seconds_after_open < execution.max_seconds_after_open < 300:
        raise ValueError("execution candidate window is invalid")
    if execution.stale_after_seconds <= 0:
        raise ValueError("stale threshold must be positive")
    if any(
        second < execution.min_seconds_after_open
        or second > execution.max_seconds_after_open
        for second in identity.fixed_evaluation_seconds
    ):
        raise ValueError("fixed evaluation seconds escape the execution window")
    split_boundaries = (
        split.fit_start,
        split.fit_end,
        split.calibration_start,
        split.calibration_end,
        split.policy_start,
        split.policy_end,
    )
    if split_boundaries != tuple(sorted(split_boundaries)):
        raise ValueError("book split boundaries must be chronological")
    if (
        split.fit_start != execution.range_start
        or split.fit_end != split.calibration_start
        or split.calibration_end != split.policy_start
        or split.policy_end != execution.range_end
    ):
        raise ValueError("book splits must be contiguous and span execution evidence")
    if config.book_evaluation is not None:
        evaluation = config.book_evaluation
        if evaluation.range_start >= evaluation.range_end:
            raise ValueError("book evaluation range must be positive")
        if evaluation.range_start < execution.range_end:
            raise ValueError(
                "book evaluation must begin after the training evidence range"
            )
        if evaluation.output_dir in {
            execution.output_dir,
            config.paths.preopen_features,
            config.paths.runs,
            config.paths.artifacts,
        }:
            raise ValueError(
                "book evaluation output must use an isolated generated path"
            )
    if not (
        0.5
        <= config.book_model.confidence_min
        <= config.book_model.confidence_max
        < 1.0
    ):
        raise ValueError("book confidence bounds must satisfy 0.5 <= min <= max < 1")
    if config.book_model.confidence_step <= 0:
        raise ValueError("book confidence step must be positive")
    probability_values = (
        config.gates.minimum_accuracy,
        config.gates.minimum_balanced_accuracy,
        config.gates.minimum_direction_recall,
        config.gates.minimum_wilson_lower_95,
        config.gates.maximum_expected_calibration_error,
        config.gates.minimum_coverage,
        config.gates.minimum_coverage_uplift,
        config.gates.maximum_accuracy_regression,
        config.gates.maximum_balanced_accuracy_regression,
        config.gates.maximum_direction_recall_regression,
    )
    if any(value < 0 or value > 1 for value in probability_values):
        raise ValueError("benchmark probability gates must be within [0, 1]")
    if config.gates.maximum_median_entry_seconds_regression >= 0:
        raise ValueError("median-entry gate must require an earlier prediction")
    if (
        config.gates.minimum_executable_markets <= 0
        or config.gates.minimum_common_time_markets <= 0
    ):
        raise ValueError("benchmark minimum sample counts must be positive")
    if config.gates.maximum_native_p99_milliseconds <= 0:
        raise ValueError("native inference latency gate must be positive")
    if config.gates.maximum_runtime_model_bytes <= 0:
        raise ValueError("runtime model byte limit must be positive")
    if (
        config.compute.max_parallel_candidates <= 0
        or config.compute.threads_per_fit <= 0
        or config.compute.polars_threads <= 0
    ):
        raise ValueError("benchmark compute limits must be positive")
    generated_paths = (
        config.paths.preopen_features,
        config.paths.runs,
        config.paths.artifacts,
    )
    if len(set(generated_paths)) != len(generated_paths):
        raise ValueError("benchmark generated paths must be isolated")


def benchmark_config_to_dict(config: EntryBenchmarkConfig) -> dict[str, Any]:
    return {
        "benchmark": {
            **asdict(config.benchmark),
            "core_config": str(config.benchmark.core_config),
        },
        "execution": {
            **asdict(config.execution),
            "range_start": config.execution.range_start.isoformat(),
            "range_end": config.execution.range_end.isoformat(),
            "output_dir": str(config.execution.output_dir),
        },
        "book_split": {
            key: value.isoformat()
            for key, value in asdict(config.book_split).items()
        },
        "book_model": asdict(config.book_model),
        "advancement_gates": asdict(config.gates),
        "compute": asdict(config.compute),
        "paths": {
            "preopen_features": str(config.paths.preopen_features),
            "runs": str(config.paths.runs),
            "artifacts": str(config.paths.artifacts),
        },
        "prior_diagnostics": (
            {
                **asdict(config.prior_diagnostics),
                "record": str(config.prior_diagnostics.record),
            }
            if config.prior_diagnostics is not None
            else None
        ),
        "book_evaluation": (
            {
                "range_start": config.book_evaluation.range_start.isoformat(),
                "range_end": config.book_evaluation.range_end.isoformat(),
                "output_dir": str(config.book_evaluation.output_dir),
            }
            if config.book_evaluation is not None
            else None
        ),
    }
