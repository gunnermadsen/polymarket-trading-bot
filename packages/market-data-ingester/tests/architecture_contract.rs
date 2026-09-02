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

fn repository_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
}
