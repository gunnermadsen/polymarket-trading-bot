use super::*;
use crate::btc::{
    directional_model::{RuntimeModelRegistry, RuntimeModelSelection},
    types::*,
};
use chrono::{DateTime, Duration, Utc};
use rust_decimal_macros::dec;
use std::{
    fs::File,
    path::{Path, PathBuf},
};
use uuid::Uuid;
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("btc-directional-model")
}
fn model(key: &str) -> std::sync::Arc<crate::btc::RuntimeDirectionalModel> {
    let dir = root().join("runtime-models").join(key);
    let m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    RuntimeModelRegistry::new(root().join("runtime-models"))
        .load(&RuntimeModelSelection {
            model_key: key.into(),
            artifact_sha256: m["model_sha256"].as_str().unwrap().into(),
            feature_schema_sha256: m["feature_schema_sha256"].as_str().unwrap().into(),
        })
        .unwrap()
}
fn book(at: DateTime<Utc>) -> OrderbookCheckpoint {
    OrderbookCheckpoint {
        checkpoint_id: Uuid::new_v4(),
        market_id: "umr-fixture".into(),
        token_id: "up".into(),
        source_timestamp: at,
        received_at: at,
        observed_at: at,
        connection_id: Uuid::nil(),
        ingest_sequence: 1,
        source_hash: None,
        tick_size: dec!(0.01),
        best_bid: Some(dec!(0.49)),
        best_ask: Some(dec!(0.5)),
        bids: vec![OrderbookLevel {
            price: dec!(0.49),
            size: dec!(1000),
        }],
        asks: vec![OrderbookLevel {
            price: dec!(0.5),
            size: dec!(1000),
        }],
        integrity_status: FeedIntegrityStatus::Ok,
    }
}
#[test]
fn frozen_core_features_match_training_from_raw_seconds() {
    use arrow_array::{Array, Int64Array, LargeStringArray, StringArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let reader = ParquetRecordBatchReaderBuilder::try_new(
        File::open(root().join("tests/fixtures/umr-core-feature-parity.parquet")).unwrap(),
    )
    .unwrap()
    .build()
    .unwrap();
    let mut checked = 0;
    for batch in reader {
        let b = batch.unwrap();
        let text = |name: &str, row: usize| {
            let a = b.column_by_name(name).unwrap();
            if let Some(a) = a.as_any().downcast_ref::<StringArray>() {
                a.value(row)
            } else {
                a.as_any()
                    .downcast_ref::<LargeStringArray>()
                    .unwrap()
                    .value(row)
            }
        };
        for row in 0..b.num_rows() {
            let seconds = b
                .column_by_name("seconds")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(row);
            let start: DateTime<Utc> = text("window_start", row).parse().unwrap();
            let klines: Vec<BinanceOneSecondKline> =
                serde_json::from_str(text("klines", row)).unwrap();
            let boundary = klines[0].open_price;
            let window = BinanceOneSecondWindow::from_completed(klines).unwrap();
            let names: Vec<String> = serde_json::from_str(text("names", row)).unwrap();
            let expected: Vec<Option<f64>> = serde_json::from_str(text("expected", row)).unwrap();
            let at = start + Duration::seconds(seconds);
            let up = book(at - Duration::milliseconds(500));
            let mut down = up.clone();
            down.token_id = "down".into();
            let oracle: Vec<crate::btc::directional_features::DirectionalOracleRound> =
                serde_json::from_str::<Vec<serde_json::Value>>(text("oracle", row))
                    .unwrap()
                    .into_iter()
                    .map(
                        |v| crate::btc::directional_features::DirectionalOracleRound {
                            phase_id: v["phase_id"].as_i64().unwrap() as i32,
                            aggregator_round_id: v["aggregator_round_id"].as_i64().unwrap(),
                            source_timestamp: v["source_timestamp"]
                                .as_str()
                                .unwrap()
                                .parse()
                                .unwrap(),
                            block_timestamp: v["block_timestamp"]
                                .as_str()
                                .unwrap()
                                .parse()
                                .unwrap(),
                            block_number: Some(v["block_number"].as_i64().unwrap()),
                            log_index: Some(0),
                            price: v["price"].as_str().unwrap().parse().unwrap(),
                            available_at: v["available_at"].as_str().unwrap().parse().unwrap(),
                        },
                    )
                    .filter(|v| v.available_at <= at)
                    .collect();
            let external = crate::btc::directional_features::DirectionalExternalFeatureInputs {
                oracle_rounds: &oracle,
                ..Default::default()
            };
            let values = crate::btc::directional_features::build_payoff_feature_values_with_policy(
                &window, start, at, boundary, &external, &up, &down, 0.02, &names, true,
            )
            .unwrap();
            for (index, (actual, expected)) in values.iter().zip(expected).enumerate() {
                match expected {
                    None => assert!(actual.is_nan(), "{} at {seconds}: {actual}", names[index]),
                    Some(v) => assert!(
                        (actual - v).abs() < 1e-7,
                        "{} at {seconds}: actual {actual}, expected {v}",
                        names[index]
                    ),
                }
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 6);
}
#[test]
fn frozen_admission_outputs_match_python_and_bindings_fail_closed() {
    let mut checked = 0;
    for entry in std::fs::read_dir(root().join("runtime-models")).unwrap() {
        let dir = entry.unwrap().path();
        let key = dir.file_name().unwrap().to_str().unwrap();
        if !key.ends_with("-umr-20260902") {
            continue;
        }
        let m = model(key);
        let adapter = m.unified_adapter().unwrap();
        let bindings = adapter
            .contract()
            .inputs
            .iter()
            .filter(|v| v.required || v.slot == "oracle")
            .map(|v| contract::SourceBinding {
                slot: v.slot.clone(),
                product: v.product.clone(),
                semantics: v.semantics.clone(),
            })
            .collect::<Vec<_>>();
        let mut binding = contract::ProcessBinding {
            version: contract::CONTRACT_VERSION.into(),
            sources: bindings,
            policy: adapter.policy().clone(),
        };
        binding.validate(adapter, 5.0).unwrap();
        assert!(binding.validate(adapter, 10.0).is_err());
        binding.sources[0].semantics = "unqualified_substitution".into();
        assert!(binding.validate(adapter, 5.0).is_err());
        let vectors: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("golden-vectors.json")).unwrap())
                .unwrap();
        for row in vectors["vectors"].as_array().unwrap() {
            let x = row["feature_values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
                .collect::<Vec<_>>();
            let actual = serde_json::to_value(
                adapter
                    .evaluate(&x, row["seconds_elapsed"].as_i64().unwrap())
                    .unwrap(),
            )
            .unwrap();
            for (name, expected) in row["source"]["admission_expected"].as_object().unwrap() {
                assert!(
                    (actual[name].as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-12,
                    "{key} {name}"
                );
            }
            assert_eq!(actual["score"]["accepted"], row["source"]["accepted"]);
            checked += 1;
        }
    }
    assert!(checked >= 250);
}
#[test]
fn causal_books_do_not_cross_epochs_or_use_future_inputs() {
    let at = DateTime::from_timestamp(Utc::now().timestamp(), 0).unwrap();
    let mut h = adapters::data::BookHistory::default();
    let old = book(at - Duration::milliseconds(500));
    h.observe(old.clone());
    let mut future = old.clone();
    future.received_at = at + Duration::milliseconds(10);
    future.source_timestamp = future.received_at;
    future.ingest_sequence = 2;
    h.observe(future);
    assert_eq!(
        h.at("umr-fixture", "up", Uuid::nil(), at)
            .unwrap()
            .ingest_sequence,
        1
    );
    assert!(h.at("umr-fixture", "up", Uuid::new_v4(), at).is_none());
    assert!(h
        .at("umr-fixture", "up", Uuid::nil(), at + Duration::seconds(4))
        .is_none());
}
#[test]
fn causal_book_boundary_survives_a_dense_live_publication_burst() {
    let at = DateTime::from_timestamp(1788816360, 0).unwrap();
    let mut history = adapters::data::BookHistory::default();
    for milliseconds in -4000..800 {
        for token in ["up", "down"] {
            let mut checkpoint = book(at + Duration::milliseconds(milliseconds));
            checkpoint.token_id = token.into();
            checkpoint.ingest_sequence = (milliseconds + 4001) as u64;
            history.observe(checkpoint);
        }
    }
    for token in ["up", "down"] {
        let selected = history.at("umr-fixture", token, Uuid::nil(), at).unwrap();
        assert_eq!(selected.received_at, at);
    }
}
#[test]
fn learned_history_recovers_automatically_without_cross_process_state() {
    let m = model("btc-5m-official-vwap-admission-umr-20260902");
    let a = m.unified_adapter().unwrap();
    let mut first = adapters::history::History::default();
    let mut other = adapters::history::History::default();
    let mut x = vec![f64::NAN; m.feature_names().len()];
    assert!(first
        .prepare("market", 65, a, &mut x, m.feature_names())
        .is_err());
    first
        .prepare("market", 60, a, &mut x, m.feature_names())
        .unwrap();
    first
        .prepare("market", 65, a, &mut x, m.feature_names())
        .unwrap();
    assert!(other
        .prepare("market", 65, a, &mut x, m.feature_names())
        .is_err());
    other
        .prepare("next-market", 60, a, &mut x, m.feature_names())
        .unwrap();
}

#[test]
fn telemetry_retains_decision_identity_and_resolves_once() {
    let id = Uuid::new_v4();
    let run = Uuid::new_v4();
    let at = Utc::now();
    let selection = RuntimeModelSelection {
        model_key: "btc-5m-umr-contract-test".into(),
        artifact_sha256: "a".repeat(64),
        feature_schema_sha256: "b".repeat(64),
    };
    telemetry::register(id, run, "test-config", "paper", Some(&selection));
    let mut snapshots = Vec::new();
    for index in 0..3 {
        let snapshot = Uuid::new_v4();
        snapshots.push(snapshot);
        telemetry::prediction(
            id,
            snapshot,
            "test-market",
            &selection,
            at + Duration::seconds(index * 5),
            &format!("{index:064x}"),
            crate::btc::RuntimeModelScore {
                raw_logit: 0.8,
                probability_up: 0.8,
                confidence: 0.8,
                action: crate::btc::RuntimeModelAction::Up,
                accepted: true,
            },
            0.001,
            None,
        );
    }
    assert_eq!(
        telemetry::prediction_record(id, snapshots[0])
            .unwrap()
            .feature_snapshot_id,
        snapshots[0]
    );
    telemetry::resolve("test-market", true);
    let first = telemetry::prometheus_metrics();
    telemetry::resolve("test-market", true);
    let scoped = |text: &str| {
        text.lines()
            .filter(|line| line.contains(&id.to_string()))
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    assert_eq!(scoped(&first), scoped(&telemetry::prometheus_metrics()));
    assert!(first.contains(&format!(
        "polymarket_umr_brier_count{{process_id=\"{id}\"}} 3"
    )));
    assert!(!first.contains("market_id="));
}

#[test]
fn ten_model_concurrent_inference_smoke() {
    let keys = [
        "btc-5m-chainlink-regime-calibrated-paper-20260820",
        "btc-5m-payoff-aware-q5-paper-20260820",
        "btc-5m-chainlink-full-combined-paper-20260820",
        "btc-5m-chainlink-stratified-payoff-paper-20260820",
        "btc-5m-specialist-distilled-fair-value-paper-20260823-v1",
        "btc-5m-extended-specialist-official-umr-20260902",
        "btc-5m-bridge-aware-specialist-umr-20260902",
        "btc-5m-official-vwap-admission-umr-20260902",
        "btc-5m-official-temporal-consensus-umr-20260902",
        "btc-5m-official-high-precision-loss-veto-umr-20260902",
    ];
    let cases = keys
        .iter()
        .map(|key| {
            let m = model(key);
            let v: serde_json::Value = serde_json::from_slice(
                &std::fs::read(
                    root()
                        .join("runtime-models")
                        .join(key)
                        .join("golden-vectors.json"),
                )
                .unwrap(),
            )
            .unwrap();
            let row = &v["vectors"][0];
            let x = row["feature_values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
                .collect::<Vec<_>>();
            (m, x, row["seconds_elapsed"].as_i64().unwrap())
        })
        .collect::<Vec<_>>();
    let start = std::time::Instant::now();
    let handles = cases
        .into_iter()
        .map(|(m, x, seconds)| {
            std::thread::spawn(move || {
                let mut durations = Vec::new();
                for _ in 0..64 {
                    let t = std::time::Instant::now();
                    let score = m.score_at_seconds(&x, seconds).unwrap();
                    assert!(score.probability_up.is_finite());
                    durations.push(t.elapsed().as_secs_f64());
                }
                durations
            })
        })
        .collect::<Vec<_>>();
    let mut times = handles
        .into_iter()
        .flat_map(|h| h.join().unwrap())
        .collect::<Vec<_>>();
    times.sort_by(f64::total_cmp);
    let report = serde_json::json!({"models":10,"inferences":times.len(),"elapsed_seconds":start.elapsed().as_secs_f64(),"p95_inference_seconds":times[times.len()*95/100],"maximum_inference_seconds":times[times.len()-1],"scope":"native host scorer concurrency; not container capacity qualification"});
    eprintln!("UMR_LOAD_EVIDENCE {report}");
    assert!(
        start.elapsed().as_secs() < 30,
        "ten-model scorer workload exceeded bounded smoke budget"
    );
}

#[test]
fn mounted_catalog_and_paper_playbooks_have_compatible_contracts() {
    let entries = catalog::discover().unwrap();
    let mut checked = 0;
    for entry in &entries {
        if !entry.model_key.ends_with("-umr-20260902") {
            continue;
        }
        assert!(entry.compatible, "{}: {:?}", entry.model_key, entry.error);
        let path = root()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("infra/processes")
            .join(format!("{}.json", entry.model_key));
        let process: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(process["enabled"], false);
        assert_eq!(process["config"]["execution"]["live_capital"], false);
        let control = &process["config"]["raw"]["btc_realtime_paper"];
        let mut strategy_value =
            serde_json::to_value(crate::btc::strategy::BtcStrategyConfig::default()).unwrap();
        strategy_value
            .as_object_mut()
            .unwrap()
            .extend(control["strategy"].as_object().unwrap().clone());
        let mut strategy: crate::btc::strategy::BtcStrategyConfig =
            serde_json::from_value(strategy_value).unwrap();
        strategy.strategy_version =
            crate::btc::directional_model::BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION.into();
        strategy.feature_schema_version = entry.feature_schema_version.clone().unwrap();
        strategy.validate().unwrap();
        let binding = strategy.unified_model.as_ref().unwrap();
        for source in &binding.sources {
            assert!(control["sources"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v.as_str() == Some(source.product.as_str())
                    || v["key"].as_str() == Some(source.product.as_str())));
        }
        let mut invalid = strategy.clone();
        invalid.target_size = dec!(10);
        assert!(invalid.validate().is_err());
        let mut invalid = strategy.clone();
        invalid.unified_model.as_mut().unwrap().policy["maximum_share_cost"] =
            serde_json::json!(0.999);
        assert!(invalid.validate().is_err());
        checked += 1;
    }
    assert_eq!(checked, 5);
    if let Ok(path) = std::env::var("UMR_VALIDATION_CATALOG_OUTPUT") {
        std::fs::write(path, serde_json::to_vec_pretty(&serde_json::json!({"contract_version":contract::CONTRACT_VERSION,"models":entries})).unwrap()).unwrap();
    }
}
