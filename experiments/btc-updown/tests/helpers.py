from btc_updown_ml.dataset import FeatureObservation, SnapshotRow


def row(
    market_index: int,
    *,
    snapshot_offset_ms: int = 0,
    feature_value: float | None = None,
    label: int | None = None,
) -> SnapshotRow:
    start = market_index * 10_000
    as_of = start + 1_000 + snapshot_offset_ms
    received = as_of
    resolved = start + 8_000
    result = SnapshotRow(
        snapshot_id=f"snapshot-{market_index}-{snapshot_offset_ms}",
        market_id=f"market-{market_index}",
        feature_as_of_ms=as_of,
        feature_received_at_ms=received,
        label_available_at_ms=resolved,
        label=market_index % 2 if label is None else label,
        task="settlement_probability_residual",
        label_version="chainlink_first_tick_at_or_after_boundary_v1",
        prior_probability=0.5,
        schema_version="features-v1",
        features=(
            FeatureObservation(
                name="distance",
                value=float(market_index if feature_value is None else feature_value),
                source_event_at_ms=as_of - 5,
                source_received_at_ms=received - 2,
            ),
        ),
    )
    result.validate()
    return result
