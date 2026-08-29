"""Probabilistic latent-state models for final TWAP settlement margin.

The implementation is deliberately CPU-only and dependency-light.  It fits
sensor calibration, transition persistence, process noise, sensor noise, and
regime transitions by chronological Gaussian maximum likelihood.  Filtering
is causal: a call consumes one market's observations in timestamp order and
never receives its completed-market target.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import norm


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    sensors: tuple[str, ...]
    regime_switching: bool


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        "twap_single_regime",
        ("twap30_margin_bps", "twap60_margin_bps"),
        False,
    ),
    CandidateSpec(
        "twap_regime_switching",
        ("twap30_margin_bps", "twap60_margin_bps"),
        True,
    ),
    CandidateSpec(
        "twap_refprice_single_regime",
        ("twap30_margin_bps", "twap60_margin_bps", "refprice_margin_bps"),
        False,
    ),
    CandidateSpec(
        "twap_refprice_regime_switching",
        ("twap30_margin_bps", "twap60_margin_bps", "refprice_margin_bps"),
        True,
    ),
)


@dataclass(frozen=True)
class SearchConfiguration:
    identifier: str
    velocity_decay: float
    process_margin_scale: float
    process_velocity_scale: float
    initial_variance_scale: float
    regime_stickiness: float


def predetermined_configurations() -> tuple[SearchConfiguration, ...]:
    """Return the frozen, identical twelve-row search budget for every arm."""

    rows = (
        (0.45, 0.50, 0.50, 0.75, 8.0),
        (0.45, 1.00, 1.00, 1.00, 8.0),
        (0.45, 2.00, 1.50, 1.50, 12.0),
        (0.65, 0.50, 1.00, 1.00, 12.0),
        (0.65, 1.00, 0.50, 1.50, 16.0),
        (0.65, 1.50, 2.00, 0.75, 8.0),
        (0.80, 0.75, 0.75, 1.25, 16.0),
        (0.80, 1.00, 1.50, 0.75, 20.0),
        (0.80, 2.00, 0.75, 1.50, 12.0),
        (0.92, 0.50, 1.50, 1.25, 20.0),
        (0.92, 1.25, 0.50, 1.00, 16.0),
        (0.92, 2.00, 2.00, 1.50, 20.0),
    )
    return tuple(
        SearchConfiguration(f"state_{index:02d}", *values)
        for index, values in enumerate(rows, start=1)
    )


@dataclass(frozen=True)
class StateSpaceParameters:
    schema_version: str
    candidate_name: str
    sensor_names: tuple[str, ...]
    sensor_intercepts: tuple[float, ...]
    sensor_loadings: tuple[float, ...]
    sensor_variances: tuple[float, ...]
    transition_phi: float
    process_margin_variance: float
    process_velocity_variance: float
    initial_margin_variance: float
    initial_velocity_variance: float
    regime_transition: tuple[tuple[float, ...], ...]
    reconstruction_variance_bps2: float
    configuration: SearchConfiguration
    fit_markets: int
    fit_rows: int
    chronological_log_likelihood: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StateSpaceParameters:
        values = dict(payload)
        values["sensor_names"] = tuple(values["sensor_names"])
        values["sensor_intercepts"] = tuple(values["sensor_intercepts"])
        values["sensor_loadings"] = tuple(values["sensor_loadings"])
        values["sensor_variances"] = tuple(values["sensor_variances"])
        values["regime_transition"] = tuple(
            tuple(row) for row in values["regime_transition"]
        )
        values["configuration"] = SearchConfiguration(**values["configuration"])
        return cls(**values)


@dataclass(frozen=True)
class ProbabilityCalibrator:
    """A direction-symmetric temperature calibrator shared across checkpoints."""

    temperature_slope: float
    support_markets: int
    support_rows: int

    def transform(self, probability_up: np.ndarray | float) -> np.ndarray:
        values = np.asarray(probability_up, dtype=float)
        clipped = np.clip(values, 1e-8, 1.0 - 1e-8)
        return expit(self.temperature_slope * logit(clipped))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProbabilityCalibrator:
        return cls(**payload)


@dataclass(frozen=True)
class FilterResult:
    probability_up: np.ndarray
    expected_margin_bps: np.ndarray
    margin_p05_bps: np.ndarray
    margin_p50_bps: np.ndarray
    margin_p95_bps: np.ndarray
    reversal_probability: np.ndarray
    margin_velocity_bps_per_step: np.ndarray
    process_uncertainty_bps2: np.ndarray
    sensor_uncertainty_bps2: np.ndarray
    regime_probabilities: np.ndarray


def class_balanced_market_weights(targets: np.ndarray) -> np.ndarray:
    labels = np.asarray(targets, dtype=float) >= 0.0
    weights = np.zeros(len(labels), dtype=float)
    for value in (False, True):
        count = int(np.count_nonzero(labels == value))
        if count:
            weights[labels == value] = 0.5 / count
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("class-balanced fitting requires at least one target")
    return weights / total


def fit_probability_calibrator(
    raw_probability_up: np.ndarray,
    labels_up: np.ndarray,
    market_ids: Iterable[str],
) -> ProbabilityCalibrator:
    probabilities = np.clip(np.asarray(raw_probability_up, dtype=float), 1e-8, 1 - 1e-8)
    labels = np.asarray(labels_up, dtype=float)
    ids = np.asarray(tuple(market_ids), dtype=object)
    if not (len(probabilities) == len(labels) == len(ids)) or not len(labels):
        raise ValueError("calibration inputs have inconsistent lengths")
    unique_ids, first = np.unique(ids, return_index=True)
    market_labels = labels[first]
    market_weights = class_balanced_market_weights(np.where(market_labels > 0.5, 1.0, -1.0))
    weight_by_id = dict(zip(unique_ids.tolist(), market_weights.tolist(), strict=True))
    counts: dict[str, int] = {}
    for market_id in ids:
        key = str(market_id)
        counts[key] = counts.get(key, 0) + 1
    row_weights = np.array(
        [weight_by_id[str(market_id)] / counts[str(market_id)] for market_id in ids],
        dtype=float,
    )
    raw_logits = logit(probabilities)

    def objective(slope: float) -> float:
        calibrated = np.clip(expit(slope * raw_logits), 1e-10, 1 - 1e-10)
        losses = -(labels * np.log(calibrated) + (1 - labels) * np.log(1 - calibrated))
        return float(np.sum(row_weights * losses))

    result = minimize_scalar(objective, bounds=(0.05, 8.0), method="bounded")
    if not result.success or not math.isfinite(float(result.x)):
        raise RuntimeError("probability calibration optimization failed")
    return ProbabilityCalibrator(float(result.x), len(unique_ids), len(labels))


def fit_chronological_mle(
    sequences: list[np.ndarray],
    targets: np.ndarray,
    candidate: CandidateSpec,
    configuration: SearchConfiguration,
    *,
    reconstruction_p99_bps: float = 0.463,
) -> StateSpaceParameters:
    """Fit a candidate from market sequences ordered oldest to newest.

    Each sequence has shape ``(checkpoints, sensors)``.  Sensor calibration is
    supervised by the final margin, while transition and regime parameters are
    estimated only from within-market chronological changes.
    """

    target_values = np.asarray(targets, dtype=float)
    if len(sequences) != len(target_values) or not len(sequences):
        raise ValueError("state-space fitting requires aligned non-empty markets")
    sensor_count = len(candidate.sensors)
    if any(sequence.ndim != 2 or sequence.shape[1] != sensor_count for sequence in sequences):
        raise ValueError("state-space sequence sensor shape does not match candidate")
    if any(len(sequence) < 2 or np.any(~np.isfinite(sequence)) for sequence in sequences):
        raise ValueError("state-space sequences must be finite with at least two checkpoints")

    market_weights = class_balanced_market_weights(target_values)
    intercepts = np.zeros(sensor_count, dtype=float)
    loadings = np.ones(sensor_count, dtype=float)
    variances = np.ones(sensor_count, dtype=float)
    reconstruction_variance = float((reconstruction_p99_bps / 2.576) ** 2)
    flattened_targets = np.concatenate(
        [np.full(len(sequence), target) for sequence, target in zip(sequences, target_values)]
    )
    flattened_weights = np.concatenate(
        [np.full(len(sequence), weight / len(sequence)) for sequence, weight in zip(sequences, market_weights)]
    )
    design_constant = np.ones(len(flattened_targets), dtype=float)
    calibrated_sequences: list[np.ndarray] = []
    for sensor in range(sensor_count):
        values = np.concatenate([sequence[:, sensor] for sequence in sequences])
        design = np.column_stack((design_constant, values))
        weighted_design = design * np.sqrt(flattened_weights)[:, None]
        weighted_target = flattened_targets * np.sqrt(flattened_weights)
        coefficient, *_ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
        intercepts[sensor] = float(coefficient[0])
        loadings[sensor] = float(np.clip(coefficient[1], -4.0, 4.0))
        residual = flattened_targets - (intercepts[sensor] + loadings[sensor] * values)
        variance = float(np.sum(flattened_weights * residual * residual))
        variances[sensor] = max(variance + reconstruction_variance, 1e-4)
    for sequence in sequences:
        calibrated_sequences.append(intercepts + sequence * loadings)

    precision = 1.0 / variances
    consensus_sequences = [
        np.sum(sequence * precision[None, :], axis=1) / precision.sum()
        for sequence in calibrated_sequences
    ]
    velocity_sequences = [np.diff(values, prepend=values[0]) for values in consensus_sequences]
    numerator = 0.0
    denominator = 0.0
    for velocity, weight in zip(velocity_sequences, market_weights):
        numerator += float(weight * np.dot(velocity[1:], velocity[:-1]))
        denominator += float(weight * np.dot(velocity[:-1], velocity[:-1]))
    fitted_phi = numerator / max(denominator, 1e-12)
    transition_phi = float(
        np.clip(0.5 * fitted_phi + 0.5 * configuration.velocity_decay, -0.95, 0.98)
    )
    margin_residuals: list[np.ndarray] = []
    velocity_residuals: list[np.ndarray] = []
    residual_weights: list[np.ndarray] = []
    for values, velocity, weight in zip(consensus_sequences, velocity_sequences, market_weights):
        margin_residuals.append(values[1:] - values[:-1] - transition_phi * velocity[:-1])
        velocity_residuals.append(velocity[1:] - transition_phi * velocity[:-1])
        residual_weights.append(np.full(len(values) - 1, weight / (len(values) - 1)))
    margin_residual = np.concatenate(margin_residuals)
    velocity_residual = np.concatenate(velocity_residuals)
    transition_weights = np.concatenate(residual_weights)
    transition_weights /= transition_weights.sum()
    process_margin_variance = max(
        float(np.sum(transition_weights * margin_residual**2))
        * configuration.process_margin_scale,
        1e-4,
    )
    process_velocity_variance = max(
        float(np.sum(transition_weights * velocity_residual**2))
        * configuration.process_velocity_scale,
        1e-4,
    )
    initial_errors = np.array(
        [values[0] - target for values, target in zip(consensus_sequences, target_values)],
        dtype=float,
    )
    initial_margin_variance = max(
        float(np.sum(market_weights * initial_errors**2))
        * configuration.initial_variance_scale,
        1e-3,
    )
    initial_velocity_variance = max(
        float(np.sum(market_weights * np.array([velocity[1] ** 2 for velocity in velocity_sequences])))
        * configuration.initial_variance_scale,
        1e-3,
    )
    regime_transition = _fit_regime_transition(
        velocity_sequences, market_weights, configuration.regime_stickiness
    )
    provisional = StateSpaceParameters(
        schema_version="btc-latent-twap-state-space-v1",
        candidate_name=candidate.name,
        sensor_names=candidate.sensors,
        sensor_intercepts=tuple(float(value) for value in intercepts),
        sensor_loadings=tuple(float(value) for value in loadings),
        sensor_variances=tuple(float(value) for value in variances),
        transition_phi=transition_phi,
        process_margin_variance=process_margin_variance,
        process_velocity_variance=process_velocity_variance,
        initial_margin_variance=initial_margin_variance,
        initial_velocity_variance=initial_velocity_variance,
        regime_transition=tuple(tuple(float(value) for value in row) for row in regime_transition),
        reconstruction_variance_bps2=reconstruction_variance,
        configuration=configuration,
        fit_markets=len(sequences),
        fit_rows=sum(len(sequence) for sequence in sequences),
        chronological_log_likelihood=0.0,
    )
    log_likelihood = 0.0
    for sequence in sequences:
        log_likelihood += _sequence_log_likelihood(sequence, candidate, provisional)
    return StateSpaceParameters(
        **{**provisional.to_dict(), "configuration": configuration, "chronological_log_likelihood": float(log_likelihood)}
    )


def filter_sequence(
    observations: np.ndarray,
    candidate: CandidateSpec,
    parameters: StateSpaceParameters,
) -> FilterResult:
    values = np.asarray(observations, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(candidate.sensors):
        raise ValueError("filter observation shape does not match candidate sensors")
    if np.any(~np.isfinite(values)):
        raise ValueError("filter observations must be finite")
    calibrated = (
        np.asarray(parameters.sensor_intercepts)[None, :]
        + values * np.asarray(parameters.sensor_loadings)[None, :]
    )
    if candidate.regime_switching:
        return _filter_regime_switching(calibrated, parameters)
    return _filter_single_regime(calibrated, parameters)


def _filter_single_regime(
    calibrated: np.ndarray, parameters: StateSpaceParameters
) -> FilterResult:
    sensor_variances = np.asarray(parameters.sensor_variances, dtype=float)
    precision = 1.0 / sensor_variances
    initial_margin = float(np.sum(calibrated[0] * precision) / precision.sum())
    state = np.array([initial_margin, 0.0], dtype=float)
    covariance = np.diag(
        [parameters.initial_margin_variance, parameters.initial_velocity_variance]
    )
    transition = np.array([[1.0, parameters.transition_phi], [0.0, parameters.transition_phi]])
    process = np.diag(
        [parameters.process_margin_variance, parameters.process_velocity_variance]
    )
    means: list[float] = []
    velocities: list[float] = []
    variances: list[float] = []
    reversals: list[float] = []
    for index, row in enumerate(calibrated):
        if index:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
        state, covariance, _ = _measurement_updates(state, covariance, row, sensor_variances)
        variance = max(float(covariance[0, 0]), 1e-9)
        sigma = math.sqrt(variance)
        crossing = float(norm.cdf(-abs(float(state[0])) / sigma))
        opposing_velocity = float(state[0] * state[1] < 0)
        reversals.append(float(np.clip(crossing + 0.20 * opposing_velocity, 0.0, 1.0)))
        means.append(float(state[0]))
        velocities.append(float(state[1]))
        variances.append(variance)
    return _filter_result(
        np.asarray(means),
        np.asarray(velocities),
        np.asarray(variances),
        np.asarray(reversals),
        np.tile(np.array([[1.0, 0.0, 0.0]]), (len(means), 1)),
        sensor_variances,
    )


def _filter_regime_switching(
    calibrated: np.ndarray, parameters: StateSpaceParameters
) -> FilterResult:
    sensor_variances = np.asarray(parameters.sensor_variances, dtype=float)
    precision = 1.0 / sensor_variances
    initial_margin = float(np.sum(calibrated[0] * precision) / precision.sum())
    states = np.tile(np.array([initial_margin, 0.0], dtype=float), (3, 1))
    covariances = np.tile(
        np.diag([parameters.initial_margin_variance, parameters.initial_velocity_variance]),
        (3, 1, 1),
    )
    probabilities = np.array([0.60, 0.30, 0.10], dtype=float)
    regime_transition = np.asarray(parameters.regime_transition, dtype=float)
    phi = parameters.transition_phi
    phis = np.array([min(abs(phi), 0.35), max(abs(phi), 0.80), -max(abs(phi), 0.55)])
    process_scales = np.array([[0.50, 0.50], [0.90, 1.20], [1.80, 2.50]])
    means: list[float] = []
    velocities: list[float] = []
    variances: list[float] = []
    reversals: list[float] = []
    regime_rows: list[np.ndarray] = []
    for index, row in enumerate(calibrated):
        if index:
            prior = probabilities @ regime_transition
            mixed_state = np.sum(probabilities[:, None] * states, axis=0)
            centered = states - mixed_state
            mixed_covariance = np.sum(
                probabilities[:, None, None]
                * (covariances + centered[:, :, None] * centered[:, None, :]),
                axis=0,
            )
            likelihoods = np.zeros(3, dtype=float)
            for regime in range(3):
                transition = np.array([[1.0, phis[regime]], [0.0, phis[regime]]])
                process = np.diag(
                    [
                        parameters.process_margin_variance * process_scales[regime, 0],
                        parameters.process_velocity_variance * process_scales[regime, 1],
                    ]
                )
                states[regime] = transition @ mixed_state
                covariances[regime] = transition @ mixed_covariance @ transition.T + process
                states[regime], covariances[regime], likelihoods[regime] = _measurement_updates(
                    states[regime], covariances[regime], row, sensor_variances
                )
            probabilities = prior * np.maximum(likelihoods, 1e-300)
            probabilities /= max(float(probabilities.sum()), 1e-300)
        else:
            for regime in range(3):
                states[regime], covariances[regime], _ = _measurement_updates(
                    states[regime], covariances[regime], row, sensor_variances
                )
        mean_state = np.sum(probabilities[:, None] * states, axis=0)
        centered = states - mean_state
        mixture_covariance = np.sum(
            probabilities[:, None, None]
            * (covariances + centered[:, :, None] * centered[:, None, :]),
            axis=0,
        )
        means.append(float(mean_state[0]))
        velocities.append(float(mean_state[1]))
        variances.append(max(float(mixture_covariance[0, 0]), 1e-9))
        reversals.append(float(probabilities[2]))
        regime_rows.append(probabilities.copy())
    return _filter_result(
        np.asarray(means),
        np.asarray(velocities),
        np.asarray(variances),
        np.asarray(reversals),
        np.asarray(regime_rows),
        sensor_variances,
    )


def _filter_result(
    means: np.ndarray,
    velocities: np.ndarray,
    variances: np.ndarray,
    reversals: np.ndarray,
    regime_probabilities: np.ndarray,
    sensor_variances: np.ndarray,
) -> FilterResult:
    sigma = np.sqrt(np.maximum(variances, 1e-9))
    probability_up = norm.cdf(means / sigma)
    return FilterResult(
        probability_up=np.asarray(probability_up, dtype=float),
        expected_margin_bps=means,
        margin_p05_bps=means + norm.ppf(0.05) * sigma,
        margin_p50_bps=means.copy(),
        margin_p95_bps=means + norm.ppf(0.95) * sigma,
        reversal_probability=reversals,
        margin_velocity_bps_per_step=velocities,
        process_uncertainty_bps2=variances,
        sensor_uncertainty_bps2=np.full(len(means), float(np.mean(sensor_variances))),
        regime_probabilities=regime_probabilities,
    )


def _measurement_updates(
    state: np.ndarray,
    covariance: np.ndarray,
    observations: np.ndarray,
    sensor_variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    updated_state = state.copy()
    updated_covariance = covariance.copy()
    log_likelihood = 0.0
    observation_vector = np.array([1.0, 0.0])
    for observation, sensor_variance in zip(observations, sensor_variances, strict=True):
        innovation = float(observation - observation_vector @ updated_state)
        innovation_variance = max(
            float(observation_vector @ updated_covariance @ observation_vector + sensor_variance),
            1e-9,
        )
        gain = updated_covariance @ observation_vector / innovation_variance
        updated_state = updated_state + gain * innovation
        updated_covariance = (
            np.eye(2) - np.outer(gain, observation_vector)
        ) @ updated_covariance
        updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
        log_likelihood += -0.5 * (
            math.log(2.0 * math.pi * innovation_variance)
            + innovation * innovation / innovation_variance
        )
    return updated_state, updated_covariance, float(math.exp(max(log_likelihood, -700.0)))


def _sequence_log_likelihood(
    observations: np.ndarray,
    candidate: CandidateSpec,
    parameters: StateSpaceParameters,
) -> float:
    calibrated = (
        np.asarray(parameters.sensor_intercepts)[None, :]
        + observations * np.asarray(parameters.sensor_loadings)[None, :]
    )
    sensor_variances = np.asarray(parameters.sensor_variances, dtype=float)
    precision = 1.0 / sensor_variances
    state = np.array([float(np.sum(calibrated[0] * precision) / precision.sum()), 0.0])
    covariance = np.diag(
        [parameters.initial_margin_variance, parameters.initial_velocity_variance]
    )
    transition = np.array(
        [[1.0, parameters.transition_phi], [0.0, parameters.transition_phi]]
    )
    process = np.diag(
        [parameters.process_margin_variance, parameters.process_velocity_variance]
    )
    total = 0.0
    for index, row in enumerate(calibrated):
        if index:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
        state, covariance, likelihood = _measurement_updates(
            state, covariance, row, sensor_variances
        )
        total += math.log(max(likelihood, 1e-300))
    return total


def _fit_regime_transition(
    velocity_sequences: list[np.ndarray],
    market_weights: np.ndarray,
    stickiness: float,
) -> np.ndarray:
    absolute = np.concatenate([np.abs(values[1:]) for values in velocity_sequences])
    stable_threshold = float(np.quantile(absolute, 0.40)) if len(absolute) else 0.0
    counts = np.ones((3, 3), dtype=float)
    counts += np.eye(3) * stickiness
    for velocity, market_weight in zip(velocity_sequences, market_weights):
        previous_sign = np.sign(velocity[:-1])
        current_sign = np.sign(velocity[1:])
        current = np.where(
            np.abs(velocity[1:]) <= stable_threshold,
            0,
            np.where((previous_sign * current_sign < 0), 2, 1),
        )
        previous = np.where(
            np.abs(velocity[:-1]) <= stable_threshold,
            0,
            np.where(
                np.r_[False, np.sign(velocity[:-2]) * previous_sign[1:] < 0],
                2,
                1,
            ),
        )
        for source, destination in zip(previous, current, strict=True):
            counts[int(source), int(destination)] += market_weight
    return counts / counts.sum(axis=1, keepdims=True)
