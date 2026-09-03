fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    std::env::set_var("PROTOC", protoc);
    let proto = "../../common/proto/market_data.proto";
    tonic_build::configure().compile_protos(&[proto], &["../../common/proto"])?;
    println!("cargo:rerun-if-changed={proto}");
    Ok(())
}
