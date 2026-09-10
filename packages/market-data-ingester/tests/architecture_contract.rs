use std::{fs, path::Path};

#[test]
fn rust_packages_use_the_root_workspace_contract() {
    let root = repository_root();
    let workspace = fs::read_to_string(root.join("Cargo.toml")).unwrap();
    assert!(workspace.contains("\"packages/market-data-ingester\""));
    assert!(workspace.contains("\"packages/polymarket-bot\""));
    assert!(workspace.contains("resolver = \"2\""));
    assert!(root.join("Cargo.lock").is_file());
    assert!(root.join("rust-toolchain.toml").is_file());
    assert!(!root.join("packages/polymarket-bot/Cargo.lock").exists());
    assert!(!root
        .join("packages/market-data-ingester/Cargo.lock")
        .exists());

    for relative in [
        "packages/polymarket-bot/Dockerfile",
        "packages/polymarket-bot/Dockerfile.production",
        "packages/market-data-ingester/Dockerfile.production",
    ] {
        let dockerfile = fs::read_to_string(root.join(relative)).unwrap();
        assert!(dockerfile.contains("COPY Cargo.toml Cargo.lock"));
        assert!(dockerfile.contains("--package "));
    }

    let ci = fs::read_to_string(root.join(".github/workflows/CI.yml")).unwrap();
    assert!(ci.contains("hashFiles('Cargo.lock')"));
    assert!(!ci.contains("packages/polymarket-bot/Cargo.lock"));
    assert!(!ci.contains("packages/market-data-ingester/Cargo.lock"));
}

#[test]
fn polymarket_image_has_no_backfill_runtime() {
    let root = repository_root();
    for relative in [
        "packages/polymarket-bot/Dockerfile",
        "packages/polymarket-bot/Dockerfile.production",
    ] {
        let dockerfile = fs::read_to_string(root.join(relative)).unwrap();
        for forbidden in [
            "polymarket-backfill-worker",
            "kraken-backfill-worker",
            "financial-data-backfill-worker",
            "backfill-plan",
        ] {
            assert!(
                !dockerfile.contains(forbidden),
                "{relative} contains {forbidden}"
            );
        }
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
fn coinapi_uses_only_the_canonical_backfill_strategy() {
    let root = repository_root();
    for legacy in [
        "scripts/materialize-coinapi-binance-spot-l2.mjs",
        "scripts/run-coinapi-binance-spot-l2-gap-backfill.mjs",
    ] {
        assert!(
            !root.join(legacy).exists(),
            "legacy CoinAPI runtime remains: {legacy}"
        );
    }
    let source = fs::read_to_string(
        root.join("packages/market-data-ingester/src/strategies/binance/coinapi_spot_l2_one_second_features_backfill.rs"),
    )
    .unwrap();
    assert_eq!(source.matches("impl BackfillWorkerStrategy for").count(), 1);
    assert!(!source.contains("RealtimeWorkerStrategy"));
    assert!(source.contains("ingester.backfill_artifacts") || source.contains("create_artifact"));
}

#[test]
fn weather_package_has_no_parallel_queue_runtime() {
    let root = repository_root();
    assert!(!root
        .join("packages/nyc-temperature-model/src/nyc_temperature_model/jobs.py")
        .exists());
    for relative in [
        "packages/nyc-temperature-model/src/nyc_temperature_model/cli.py",
        "packages/nyc-temperature-model/src/nyc_temperature_model/config.py",
    ] {
        let source = fs::read_to_string(root.join(relative)).unwrap();
        for forbidden in [
            "weather.ingestion_jobs",
            "WEATHER_WORKER_INGESTERS",
            "run_worker",
        ] {
            assert!(
                !source.contains(forbidden),
                "{relative} contains {forbidden}"
            );
        }
    }
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
        assert!(compose.contains("capitonic/ingester:"));
        assert!(!compose.contains("capitonic/ingester-master:"));
        assert!(!compose.contains("capitonic/ingester-worker:"));
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
fn binance_l2_persistence_alert_returns_a_healthy_zero() {
    let root = repository_root();
    let rules = fs::read_to_string(
        root.join("common/configs/grafana/provisioning/alerting/rules-market-data-ingester.yml"),
    )
    .unwrap();
    let rule = rules
        .split("- uid: mdi_binance_l2_persistence_stale")
        .nth(1)
        .and_then(|remaining| remaining.split("- uid:").next())
        .expect("Binance L2 persistence alert is provisioned");

    assert!(rule.contains(") > bool 60)"));
    assert!(rule.contains("noDataState: OK"));
}

#[test]
fn polymarket_chainlink_twap_critical_alert_requires_repeated_transport_gaps() {
    let root = repository_root();
    let rules = fs::read_to_string(
        root.join("common/configs/grafana/provisioning/alerting/rules-market-data-ingester.yml"),
    )
    .unwrap();
    let rule = rules
        .split("- uid: mdi_critical_11")
        .nth(1)
        .and_then(|remaining| remaining.split("- uid:").next())
        .expect("Polymarket Chainlink TWAP critical alert is provisioned");

    assert!(rule.contains("polymarket_rtds_thirty_stale|polymarket_rtds_sixty_stale"));
    assert!(rule.contains("error_code=\"polymarket_twap_transport_gap\""));
    assert!(rule.contains("[5m])) >= 3)"));
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
    let dockerfile = fs::read_to_string(root.join("Dockerfile.production")).unwrap();
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

#[test]
fn economic_backfills_use_one_unified_strategy_per_file() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for relative in [
        "src/strategies/economic/fred_economic_series_backfill.rs",
        "src/strategies/economic/new_york_fed_reference_rates_backfill.rs",
        "src/strategies/economic/new_york_fed_soma_holdings_backfill.rs",
        "src/strategies/economic/cftc_legacy_futures_backfill.rs",
        "src/strategies/economic/cftc_traders_financial_futures_backfill.rs",
        "src/strategies/treasury/us_treasury_auctions_backfill.rs",
        "src/strategies/treasury/us_treasury_debt_to_penny_backfill.rs",
        "src/strategies/treasury/us_treasury_deposits_withdrawals_backfill.rs",
        "src/strategies/treasury/us_treasury_operating_cash_balance_backfill.rs",
    ] {
        let source = fs::read_to_string(root.join(relative)).unwrap();
        assert_eq!(
            source.matches("define_economic_strategy!").count()
                + source.matches("define_treasury_strategy!").count(),
            1,
            "{relative} must define exactly one strategy"
        );
        assert!(!source.contains("RealtimeWorkerStrategy"), "{relative}");
    }

    let support =
        fs::read_to_string(root.join("src/strategies/economic/backfill_support.rs")).unwrap();
    for forbidden in [
        "financial_data.backfill_jobs",
        "financial_data.worker_status",
        "financial_data.backfill_job_events",
    ] {
        assert!(
            !support.contains(forbidden),
            "economic strategies reference legacy queue state: {forbidden}"
        );
    }
}

#[test]
fn market_data_contracts_have_one_authoritative_definition() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let domain = fs::read_to_string(root.join("src/domain/dataset.rs")).unwrap();
    let bindings = fs::read_to_string(root.join("src/strategies/datasets.rs")).unwrap();
    let strategy_root = root.join("src/strategies");

    for table in [
        "market_data.binance_spot_btcusdt_aggregate_trades",
        "market_data.binance_spot_btcusdt_one_second_ohlcv",
        "market_data.binance_futures_btcusdt_open_interest",
        "market_data.chainlink_btcusd_reference_prices",
        "market_data.chainlink_btcusd_one_minute_candles",
        "market_data.polygon_chainlink_btcusd_oracle_rounds",
        "market_data.pmdata_chainlink_btcusd_twap",
    ] {
        assert!(domain.contains(table), "canonical contract omits {table}");
    }

    for pair in [
        [
            "binance_spot_btcusdt_aggregate_trades",
            "binance_spot_btcusdt_aggregate_trades_backfill",
        ],
        [
            "binance_spot_btcusdt_one_second_ohlcv",
            "binance_spot_btcusdt_one_second_ohlcv_backfill",
        ],
        [
            "binance_futures_btcusdt_open_interest",
            "binance_futures_btcusdt_five_minute_open_interest_backfill",
        ],
        [
            "chainlink_btcusd_reference_price",
            "chainlink_btcusd_reference_ticks_backfill",
        ],
        [
            "chainlink_btcusd_one_minute_ohlc",
            "chainlink_btcusd_one_minute_candles_backfill",
        ],
        [
            "polygon_chainlink_btcusd_oracle",
            "polygon_chainlink_btcusd_oracle_rounds_backfill",
        ],
    ] {
        for strategy in pair {
            assert!(
                bindings.contains(strategy),
                "dataset binding omits {strategy}"
            );
        }
    }

    for relative in [
        "binance/types.rs",
        "chainlink/backfill_types.rs",
        "polygon/backfill_types.rs",
        "pmdata/types.rs",
    ] {
        let source = fs::read_to_string(strategy_root.join(relative)).unwrap();
        for canonical in [
            "BinanceAggregateTradeRecord",
            "BinanceOneSecondKlineRecord",
            "BinanceBtcusdtOpenInterestRecord",
            "ChainlinkBtcusdArchiveTick",
            "ChainlinkBtcusdOneMinuteCandle",
            "PolygonChainlinkBtcusdOracleRound",
            "PmdataChainlinkBtcusdTwapRecord",
            "PmdataChainlinkBtcusdRefpriceRecord",
        ] {
            assert!(
                !source.contains(&format!("struct {canonical}")),
                "{relative} redefines canonical record {canonical}"
            );
        }
    }
}

#[test]
fn binance_open_interest_strategies_share_the_canonical_table() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let realtime =
        fs::read_to_string(root.join("src/strategies/binance/futures_open_interest.rs")).unwrap();
    let backfill = fs::read_to_string(
        root.join("src/strategies/binance/five_minute_open_interest_backfill.rs"),
    )
    .unwrap();
    let canonical = "market_data.binance_futures_btcusdt_open_interest";
    let legacy = "polymarket.binance_btcusdt_five_minute_open_interest";

    assert!(realtime.contains(canonical));
    assert!(backfill.contains(canonical));
    assert!(!realtime.contains(legacy));
    assert!(!backfill.contains(legacy));
}

#[test]
fn polymarket_orderbooks_have_one_final_physical_table_contract() {
    let root = repository_root();
    let dataset =
        fs::read_to_string(root.join("packages/market-data-ingester/src/domain/dataset.rs"))
            .unwrap();
    let strategy = fs::read_to_string(
        root.join("packages/market-data-ingester/src/strategies/polymarket/orderbook_snapshots.rs"),
    )
    .unwrap();
    let final_table = "polymarket.btc_five_minute_orderbook_snapshots";
    let retired_table = "market_data.polymarket_btc_five_minute_orderbook_snapshots";
    assert!(dataset.contains(final_table));
    assert!(strategy.contains(final_table));
    assert!(!dataset.contains(retired_table));
    assert!(!strategy.contains(retired_table));

    let migration = fs::read_to_string(root.join(
        "packages/db-migrate/src/migrations/1788646000000-FinalizePolymarketOrderbookStorage.ts",
    ))
    .unwrap();
    let cleanup = fs::read_to_string(root.join(
        "packages/db-migrate/src/migrations/1788646001000-DropLegacyPolymarketOrderbookStorage.ts",
    ))
    .unwrap();
    assert!(cleanup.contains("DROP TABLE ${LEGACY}"));
    assert!(cleanup.contains("drop_chunks"));
    assert!(migration.contains("must be fully stopped before table cutover"));
    assert!(migration.contains("advanced beyond archived watermark"));
}

#[test]
fn chainlink_reference_products_have_exclusive_persistence_boundaries() {
    let root = repository_root();
    let persistence = fs::read_to_string(
        root.join("packages/market-data-ingester/src/persistence/chainlink_reference_prices.rs"),
    )
    .unwrap();
    assert!(persistence.contains("market_data.chainlink_btcusd_reference_prices"));
    assert!(persistence.contains("market_data.pmdata_chainlink_btcusd_reference_prices"));

    for relative in [
        "packages/market-data-ingester/src/strategies/chainlink/reference_price.rs",
        "packages/market-data-ingester/src/strategies/chainlink/reference_ticks_backfill.rs",
        "packages/market-data-ingester/src/strategies/pmdata/backfill_runtime.rs",
    ] {
        let source = fs::read_to_string(root.join(relative)).unwrap();
        assert!(!source.contains("INSERT INTO market_data.chainlink_btcusd_reference_prices"));
        assert!(
            !source.contains("INSERT INTO market_data.pmdata_chainlink_btcusd_reference_prices")
        );
        assert!(!source.contains("INSERT INTO polymarket.chainlink_btcusd_archive_ticks"));
    }

    let bindings =
        fs::read_to_string(root.join("packages/market-data-ingester/src/strategies/datasets.rs"))
            .unwrap();
    assert!(bindings.contains("DatasetKey::PmdataChainlinkBtcusdReferencePrices"));
}

#[test]
fn aggregate_trade_persistence_has_one_repository_and_no_legacy_runtime_table() {
    let root = repository_root();
    let repository = fs::read_to_string(
        root.join("packages/market-data-ingester/src/persistence/aggregate_trades.rs"),
    )
    .unwrap();
    assert!(repository.contains("market_data.binance_spot_btcusdt_aggregate_trades"));

    for relative in [
        "packages/market-data-ingester/src/strategies/binance/aggregate_trades.rs",
        "packages/market-data-ingester/src/strategies/binance/aggregate_trades_backfill.rs",
        "packages/btc-directional-model/sql/btc-binance-trade-print-source.sql",
        "packages/btc-directional-model/sql/btc-refprice-context-trade-print-source.sql",
    ] {
        let source = fs::read_to_string(root.join(relative)).unwrap();
        assert!(
            !source.contains("polymarket.binance_aggregate_trades"),
            "legacy aggregate-trade relation remains in {relative}"
        );
        if relative.contains("strategies/binance/aggregate_trades") {
            assert!(
                !source.contains("INSERT INTO market_data.binance_spot_btcusdt_aggregate_trades"),
                "strategy bypasses the canonical aggregate-trade repository: {relative}"
            );
        }
    }
}

#[test]
fn binance_l2_feature_strategies_have_one_destination_per_market() {
    let root = repository_root();
    let spot = "market_data.binance_spot_btcusdt_l2_one_second_features";
    let futures = "market_data.binance_futures_btcusdt_l2_one_second_features";
    let files = [
        (
            "packages/market-data-ingester/src/strategies/binance/spot_l2_one_second_features_backfill.rs",
            spot,
        ),
        (
            "packages/market-data-ingester/src/strategies/binance/coinapi_spot_l2_one_second_features_backfill.rs",
            spot,
        ),
        (
            "packages/market-data-ingester/src/strategies/binance/futures_l2_one_second_features_backfill.rs",
            futures,
        ),
    ];
    for (relative, canonical) in files {
        let source = fs::read_to_string(root.join(relative)).unwrap();
        assert!(source.contains(canonical), "{relative} omits {canonical}");
        assert!(!source.contains("polymarket.binance_spot_btcusdt_l2_one_second_features"));
        assert!(!source.contains("polymarket.binance_btcusdt_l2_one_second_features"));
        assert!(!source.contains("_staging"));
    }
    let persistence = fs::read_to_string(
        root.join("packages/market-data-ingester/src/strategies/binance/l2_backfill_support.rs"),
    )
    .unwrap();
    assert!(!persistence.contains("_staging"));
    assert!(persistence.contains("ON CONFLICT (symbol,second_start) DO NOTHING"));
}

#[test]
fn polymarket_execution_snapshots_have_one_physical_write_destination() {
    let root = repository_root();
    let support = fs::read_to_string(
        root.join("packages/market-data-ingester/src/strategies/polymarket/backfill/support.rs"),
    )
    .unwrap();
    assert!(support.contains("INSERT INTO polymarket.btc_market_capacity_execution_snapshots"));
    assert!(!support.contains("INSERT INTO polymarket.btc_market_execution_snapshots"));
    assert!(!support.contains("INSERT INTO polymarket.btc_market_decision_execution_snapshots"));
}

#[test]
fn binance_backfills_use_the_authoritative_support_module() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert!(root.join("src/strategies/backfill_support.rs").exists());
    assert!(
        !root
            .join("src/strategies/binance/backfill_support.rs")
            .exists(),
        "Binance must not define a parallel backfill support module"
    );

    let module = fs::read_to_string(root.join("src/strategies/binance/mod.rs")).unwrap();
    assert!(!module.contains("mod backfill_support;"));

    let binance = root.join("src/strategies/binance");
    for entry in fs::read_dir(binance).unwrap().filter_map(Result::ok) {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.ends_with("backfill.rs") {
            continue;
        }
        let source = fs::read_to_string(&path).unwrap();
        for forbidden in [
            "async fn require_lease(",
            "fn database_error(error: sqlx::Error)",
            "fn validate_empty_request(",
        ] {
            assert!(
                !source.contains(forbidden),
                "{name} duplicates authoritative support: {forbidden}"
            );
        }
    }
}

fn repository_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
}
