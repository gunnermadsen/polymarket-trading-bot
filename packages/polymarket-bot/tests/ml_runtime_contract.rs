use polymarket_bot::ml::{
    score_linear_logit, sha256_hex, LinearFeatureTransform, LinearLogitArtifact, MlFeature,
    MlFeatureVector, MlTask, PriorKind,
};
use serde_json::Value;

const FIXTURE: &str =
    include_str!("../../../experiments/btc-updown/fixtures/runtime_v2_contract.json");

fn task(value: &str) -> MlTask {
    match value {
        "settlement_probability_residual" => MlTask::SettlementProbabilityResidual,
        "fok_fill_probability" => MlTask::FokFillProbability,
        "fok_post_fill_toxicity_probability" => MlTask::FokPostFillToxicityProbability,
        unsupported => panic!("unsupported fixture task {unsupported}"),
    }
}

#[test]
fn runtime_v2_contract_matches_python_fixture() {
    let fixture: Value = serde_json::from_str(FIXTURE).expect("fixture must be valid JSON");
    assert_eq!(fixture["contract_version"], "btc_5m_ml_runtime_contract_v2");
    let manifest_seed = fixture["manifest"]["seed"].as_str().unwrap();
    let manifest_sha256 = fixture["manifest"]["sha256"].as_str().unwrap();
    assert_eq!(sha256_hex(manifest_seed), manifest_sha256);

    let mut settlement_artifact = None;
    for specification in fixture["artifacts"].as_array().unwrap() {
        let task = task(specification["task"].as_str().unwrap());
        let transforms = specification["feature_names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| LinearFeatureTransform::new(name.as_str().unwrap(), 0.0, 1.0, 0.0).unwrap())
            .collect::<Vec<_>>();
        let artifact = LinearLogitArtifact::new(
            specification["model_version"].as_str().unwrap(),
            task,
            PriorKind::ProvidedProbability,
            specification["feature_schema_version"].as_str().unwrap(),
            Some(manifest_sha256.to_string()),
            None,
            0.0,
            transforms,
        )
        .unwrap();
        assert_eq!(
            artifact.feature_schema_sha256(),
            specification["expected_feature_schema_sha256"]
                .as_str()
                .unwrap()
        );
        assert_eq!(
            artifact.artifact_sha256(),
            specification["expected_artifact_sha256"].as_str().unwrap()
        );
        if task == MlTask::SettlementProbabilityResidual {
            settlement_artifact = Some(artifact);
        }
    }

    let specification = &fixture["vector"];
    let features = specification["features"]
        .as_array()
        .unwrap()
        .iter()
        .map(|feature| {
            MlFeature::new(
                feature["name"].as_str().unwrap(),
                feature["value"].as_f64().unwrap(),
                feature["source_event_at_ms"].as_i64().unwrap(),
                feature["source_received_at_ms"].as_i64().unwrap(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let vector = MlFeatureVector::new(
        specification["snapshot_id"].as_str().unwrap(),
        specification["market_id"].as_str().unwrap(),
        specification["feature_as_of_ms"].as_i64().unwrap(),
        specification["feature_received_at_ms"].as_i64().unwrap(),
        specification["schema_version"].as_str().unwrap(),
        specification["prior_probability"].as_f64().unwrap(),
        features,
    )
    .unwrap();
    assert_eq!(
        vector.schema_sha256(),
        specification["expected_feature_schema_sha256"]
            .as_str()
            .unwrap()
    );
    assert_eq!(
        vector.vector_sha256(),
        specification["expected_vector_sha256"].as_str().unwrap()
    );
    let score = score_linear_logit(&settlement_artifact.unwrap(), &vector).unwrap();
    assert_eq!(
        score.probability(),
        specification["expected_probability"].as_f64().unwrap()
    );
    assert_eq!(
        score.residual_logit(),
        specification["expected_residual_logit"].as_f64().unwrap()
    );
}
