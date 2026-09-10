use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

fn collect_files(directory: &Path, files: &mut Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_files(&path, files)?;
        } else if path.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-env-changed=POLYMARKET_BUILD_SOURCE_ID");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let workspace_dir = manifest_dir
        .parent()
        .and_then(Path::parent)
        .ok_or("package manifest is not beneath the workspace root")?;
    let mut files = vec![
        workspace_dir.join("Cargo.toml"),
        workspace_dir.join("Cargo.lock"),
        manifest_dir.join("Cargo.toml"),
        manifest_dir.join("build.rs"),
    ];
    collect_files(&manifest_dir.join("src"), &mut files)?;
    files.sort();

    let mut hasher = Sha256::new();
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path.strip_prefix(workspace_dir)?.to_string_lossy();
        let contents = fs::read(&path)?;
        hasher.update((relative.len() as u64).to_le_bytes());
        hasher.update(relative.as_bytes());
        hasher.update((contents.len() as u64).to_le_bytes());
        hasher.update(contents);
    }
    let tree_hash = format!("{:x}", hasher.finalize());
    let external_identity = env::var("POLYMARKET_BUILD_SOURCE_ID")
        .ok()
        .map(|value| value.replace(['\r', '\n'], ""))
        .filter(|value| !value.trim().is_empty());
    let identity = external_identity.map_or_else(
        || format!("tree-sha256:{tree_hash}"),
        |external| format!("{external}+tree-sha256:{tree_hash}"),
    );
    println!("cargo:rustc-env=POLYMARKET_COMPILED_SOURCE_ID={identity}");
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    env::set_var("PROTOC", protoc);
    let proto = "../../common/proto/market_data.proto";
    tonic_build::configure().compile_protos(&[proto], &["../../common/proto"])?;
    println!("cargo:rerun-if-changed={proto}");
    Ok(())
}
