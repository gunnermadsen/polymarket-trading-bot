from copy import deepcopy
from datetime import UTC, date, datetime

import pytest

from nyc_temperature_model.residual_opportunity_benchmark import (
    _canonicalize_fit_records,
    _finalize_scored_candidate_ledger,
    _retrospective_screen_supported,
    _runtime_provenance,
    _source_candidate_digest,
    _validate_event_partition_label,
    _validate_source_candidate_invariants,
)


def _candidate(hour: int, side: str) -> dict:
    resolved = side == "YES"
    ask = 0.09
    all_in = 0.10
    model_id = f"model-{hour}"
    decision_time = datetime(2026, 7, 1, 4 if hour == 0 else 16, tzinfo=UTC)
    return {
        "process_id": "process",
        "model_run_id": model_id,
        "market_id": f"market-{hour}",
        "event_date": date(2026, 7, 1),
        "decision_time": decision_time,
        "decision_hour_local": hour,
        "side": side,
        "quantity": 5.0,
        "ask_vwap": ask,
        "fees_enabled": False,
        "fee_rate": 0.0,
        "fee_exponent": 1.0,
        "fee_taker_only": True,
        "fee_per_share": 0.0,
        "modeled_slippage_per_share": 0.01,
        "all_in_cost_per_share": all_in,
        "break_even_probability": all_in,
        "resolved_side": resolved,
        "realized_net_per_share": (1.0 if resolved else 0.0) - all_in,
        "executable": True,
        "rejection_reasons": [],
        "eligible": False,
        "selected": False,
    }


def _source_run() -> dict:
    return {"decision_models": {"0": "model-0", "12": "model-12"}}


def test_runtime_provenance_requires_an_explicit_runner_inspected_image_id(monkeypatch):
    monkeypatch.setenv("POLYMARKET_GIT_REVISION", "a" * 40)
    monkeypatch.setenv("WEATHER_MODEL_IMAGE_DIGEST", f"sha256:{'c' * 64}")
    image_id = f"sha256:{'b' * 64}"

    provenance = _runtime_provenance(image_id)

    assert provenance["git_revision"] == "a" * 40
    assert provenance["runner_declared_weather_model_image_id"] == image_id
    assert "docker image inspect" in provenance["weather_model_image_id_source_contract"]
    with pytest.raises(ValueError, match="weather model image ID"):
        _runtime_provenance("")


def test_source_candidate_economics_and_hour_model_contract_are_fail_fast():
    candidates = [
        _candidate(hour, side) for hour in (0, 12) for side in ("YES", "NO")
    ]

    _validate_source_candidate_invariants(candidates, _source_run())

    mutations = (
        ("quantity", 1.0, "non-five-share quantity"),
        ("modeled_slippage_per_share", 0.0, "one-cent slippage"),
        ("fee_per_share", 0.01, "captured fee does not recompute"),
        ("all_in_cost_per_share", 0.11, "all_in_cost_per_share does not recompute"),
        ("realized_net_per_share", 0.0, "realized PnL does not recompute"),
    )
    for field, value, message in mutations:
        altered = deepcopy(candidates)
        altered[0][field] = value
        with pytest.raises(ValueError, match=message):
            _validate_source_candidate_invariants(altered, _source_run())

    altered = deepcopy(candidates)
    altered[0]["model_run_id"] = "wrong-model"
    with pytest.raises(ValueError, match="hour/model mapping"):
        _validate_source_candidate_invariants(altered, _source_run())


def test_event_partition_requires_all_buckets_resolved_and_exactly_one_winner():
    event_date = date(2026, 7, 1)
    row = {
        "market_id": "market-0",
        "event_id": "event-0",
        "event_date": event_date,
        "canonical_event_date": event_date,
        "event_partition_min_date": event_date,
        "event_partition_max_date": event_date,
        "event_partition_label_complete": True,
        "event_partition_bucket_count": 7,
        "event_partition_winner_count": 1,
        "label_available_at": datetime(2026, 7, 2, tzinfo=UTC),
        "side": "YES",
        "canonical_resolved_yes": True,
        "resolved_side": True,
    }

    _validate_event_partition_label(row)

    incomplete = {**row, "event_partition_label_complete": False}
    with pytest.raises(ValueError, match="complete resolution label"):
        _validate_event_partition_label(incomplete)
    multiple_winners = {**row, "event_partition_winner_count": 2}
    with pytest.raises(ValueError, match="exactly one winning bucket"):
        _validate_event_partition_label(multiple_winners)


def test_fit_records_and_candidate_references_have_deterministic_identities():
    origin = datetime(2026, 7, 1, tzinfo=UTC)
    core = {
        "origin_time": origin,
        "latest_label_available_at": datetime(2026, 6, 30, tzinfo=UTC),
        "training_start": date(2026, 4, 1),
        "training_end": date(2026, 6, 29),
        "training_event_days": 50,
        "training_rows": 100,
        "coefficients": {
            "midnight_weather_residual_weight": 0.1,
            "noon_weather_residual_weight": 0.2,
        },
    }
    first_candidates = [{"opportunity_fit_id": "random-a"}]
    second_candidates = [{"opportunity_fit_id": "random-b"}]

    first = _canonicalize_fit_records(
        first_candidates,
        [{"opportunity_fit_id": "random-a", **core}],
        source_policy_run_id="source-run",
        source_candidate_sha256="a" * 64,
    )
    second = _canonicalize_fit_records(
        second_candidates,
        [{"opportunity_fit_id": "random-b", **core}],
        source_policy_run_id="source-run",
        source_candidate_sha256="a" * 64,
    )

    assert first == second
    assert first_candidates[0]["opportunity_fit_id"] == first[0]["opportunity_fit_id"]
    assert second_candidates[0]["opportunity_fit_id"] == first[0]["opportunity_fit_id"]
    assert len(first[0]["fit_record_sha256"]) == 64


def test_full_scored_ledger_disposition_and_digest_are_stable():
    selected = _candidate(0, "YES")
    rejected = _candidate(12, "NO")
    rows = [selected, rejected]
    selected_key = (
        selected["model_run_id"],
        selected["market_id"],
        selected["decision_time"],
        selected["side"],
    )
    rejected_key = (
        rejected["model_run_id"],
        rejected["market_id"],
        rejected["decision_time"],
        rejected["side"],
    )

    _finalize_scored_candidate_ledger(
        rows,
        selected=[selected],
        rejection_reasons={selected_key: [], rejected_key: ["robust_edge_below_policy"]},
    )

    assert selected["eligible"] and selected["selected"]
    assert not rejected["eligible"] and not rejected["selected"]
    assert _source_candidate_digest(rows) == _source_candidate_digest(deepcopy(rows))


def test_retrospective_screen_requires_tail_and_best_trade_robustness():
    checks = {
        "evaluation_positive_total_net": True,
        "evaluation_positive_return_on_capital": True,
        "evaluation_positive_lower_90pct_daily_net": True,
        "evaluation_positive_without_best_trade": True,
        "evaluation_net_exceeds_weather_only": True,
    }

    assert _retrospective_screen_supported(checks)
    for required in (
        "evaluation_positive_lower_90pct_daily_net",
        "evaluation_positive_without_best_trade",
    ):
        altered = {**checks, required: False}
        assert not _retrospective_screen_supported(altered)
