define_kraken_futures_strategy!(
    KrakenLiquidationVolumeBackfill,
    "kraken_liquidation_volume_backfill",
    "Kraken liquidation volume backfill",
    "Collects historical Kraken Futures liquidation volume analytics",
    LiquidationVolume,
    time
);
