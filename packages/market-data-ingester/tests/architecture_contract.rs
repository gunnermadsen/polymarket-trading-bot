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
}

#[test]
fn compose_exposes_only_standard_ingester_roles() {
    let root = repository_root();
    let base = fs::read_to_string(root.join("docker-compose.yml")).unwrap();
    let ingester =
        fs::read_to_string(root.join("packages/market-data-ingester/docker-compose.yml")).unwrap();
    for forbidden in [
        "polymarket-backfill-worker:",
        "kraken-backfill-worker-1:",
        "financial-data-backfill-worker-1:",
        "pmdata-backfill-worker-1:",
    ] {
        assert!(
            !base.contains(forbidden),
            "legacy Compose service remains: {forbidden}"
        );
    }
    assert!(ingester.contains("ingester-master:"));
    assert!(ingester.contains("ingester-worker:"));
}

fn repository_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
}
