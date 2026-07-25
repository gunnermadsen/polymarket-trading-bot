use std::io;

use anyhow::Context;
use polymarket_bot::btc::reliability_calibration::{
    build_anchored_dataset, fit, CalibrationInputRow,
};

fn main() -> anyhow::Result<()> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .flexible(false)
        .from_reader(io::stdin());
    let rows = reader
        .deserialize::<CalibrationInputRow>()
        .collect::<Result<Vec<_>, _>>()
        .context("decode tab-separated calibration rows from stdin")?;
    let samples = build_anchored_dataset(rows)?;
    let report = fit(&samples)?;
    serde_json::to_writer_pretty(io::stdout(), &report)?;
    println!();
    anyhow::ensure!(
        report.holdout_accepted,
        "frozen calibration rejected: holdout did not improve both log loss and Brier score"
    );
    Ok(())
}
