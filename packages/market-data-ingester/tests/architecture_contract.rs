use std::{fs, path::Path};

#[test]
fn polymarket_image_has_no_backfill_runtime() {
    let root = repository_root();
    let dockerfile = fs::read_to_string(root.join("packages/polymarket-bot/Dockerfile")).unwrap();
    for forbidden in [
        "polymarket-backfill-worker",
        "kraken-backfill-worker",
        "financial-data-backfill-worker",
        "backfill-plan",
    ] {
        assert!(
            !dockerfile.contains(forbidden),
            "Polymarket image contains {forbidden}"
        );
    }
    let binary_dir = root.join("packages/polymarket-bot/src/bin");
    let legacy_binaries = fs::read_dir(binary_dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.contains("backfill"))
        .collect::<Vec<_>>();
    assert!(
        legacy_binaries.is_empty(),
        "Polymarket crate still exposes backfill binaries: {legacy_binaries:?}"
    );
}

#[test]
fn compose_exposes_only_standard_ingester_roles() {
    let root = repository_root();
    let base = fs::read_to_string(root.join("docker-compose.yml")).unwrap();
    let production = fs::read_to_string(root.join("docker-compose.production.yml")).unwrap();
    for forbidden in [
        "polymarket-backfill-worker:",
        "kraken-backfill-worker-1:",
        "financial-data-backfill-worker-1:",
        "pmdata-backfill-worker-1:",
    ] {
        for compose in [&base, &production] {
            assert!(
                !compose.contains(forbidden),
                "legacy Compose service remains: {forbidden}"
            );
        }
    }
    for compose in [&base, &production] {
        assert!(compose.contains("ingester-master:"));
        assert!(compose.contains("ingester-worker:"));
    }
    let compose_files = fs::read_dir(root)
        .unwrap()
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.starts_with("docker-compose") && name.ends_with(".yml"))
        .collect::<Vec<_>>();
    assert_eq!(
        compose_files.len(),
        2,
        "unexpected Compose files: {compose_files:?}"
    );
    assert!(!root
        .join("packages/market-data-ingester/docker-compose.yml")
        .exists());
}

#[test]
fn kraken_backfills_are_one_strategy_per_file() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/strategies/kraken");
    for file in [
        "instruments_backfill.rs",
        "fee_schedules_backfill.rs",
        "trade_candles_backfill.rs",
        "mark_candles_backfill.rs",
        "spot_candles_backfill.rs",
        "open_interest_backfill.rs",
        "future_basis_backfill.rs",
        "aggressor_differential_backfill.rs",
        "trade_volume_backfill.rs",
        "trade_count_backfill.rs",
        "cvd_backfill.rs",
        "liquidation_volume_backfill.rs",
        "spreads_backfill.rs",
        "liquidity_backfill.rs",
        "slippage_backfill.rs",
        "funding_rates_backfill.rs",
        "spot_trade_prints_one_second_ohlcv_backfill.rs",
    ] {
        let source = std::fs::read_to_string(root.join(file)).unwrap();
        assert_eq!(
            source.matches("impl BackfillWorkerStrategy for").count()
                + source.matches("define_kraken_futures_strategy!").count(),
            1,
            "{file} must define exactly one backfill strategy"
        );
        assert!(
            !source.contains("RealtimeWorkerStrategy"),
            "{file} must remain backfill-only"
        );
    }
}

#[test]
fn raw_weather_backfills_are_native_and_one_strategy_per_file() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for relative in [
        "src/strategies/weather/backfill/goes_abi_source_archives.rs",
        "src/strategies/weather/backfill/hrrr_surface_archives.rs",
        "src/strategies/weather/backfill/asos_one_minute_archives.rs",
        "src/strategies/weather/backfill/asos_metar_archives.rs",
        "src/strategies/temperature/backfill/polymarket_market_archives.rs",
        "src/strategies/temperature/backfill/polymarket_price_archives.rs",
        "src/strategies/temperature/backfill/pmxt_orderbook_archives.rs",
    ] {
        let source = fs::read_to_string(root.join(relative)).unwrap();
        assert_eq!(
            source.matches("impl BackfillWorkerStrategy for").count(),
            1,
            "{relative}"
        );
        assert!(!source.contains("RealtimeWorkerStrategy"), "{relative}");
    }
    let strategies = fs::read_to_string(root.join("src/strategies/mod.rs")).unwrap();
    for forbidden in ["nyc-temperature-model", "Command::new", "python"] {
        assert!(
            !strategies.contains(forbidden),
            "strategy registry contains {forbidden}"
        );
    }
    let dockerfile = fs::read_to_string(root.join("Dockerfile")).unwrap();
    for forbidden in [
        "FROM python:",
        "pip install",
        "packages/nyc-temperature-model",
    ] {
        assert!(
            !dockerfile.contains(forbidden),
            "ingester image contains {forbidden}"
        );
    }
}

fn repository_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
}
