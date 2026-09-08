//! Process-owned admission history. Unknown pre-restart opportunities cannot be
//! fabricated. Readiness automatically recovers at the next market's first slot.
use super::ModelAdapter;
use anyhow::{ensure, Result};
#[derive(Default)]
pub struct History {
    market: String,
    values: Vec<(i64, f64)>,
    observed: Vec<i64>,
}
impl History {
    pub fn observe_slot(&mut self, market: &str, seconds: i64) {
        if self.market != market {
            self.market = market.into();
            self.values.clear();
            self.observed.clear();
        }
        if (60..=85).contains(&seconds) && seconds % 5 == 0 && !self.observed.contains(&seconds) {
            self.observed.push(seconds);
        }
    }
    pub fn prepare(
        &mut self,
        market: &str,
        seconds: i64,
        adapter: &dyn ModelAdapter,
        features: &mut [f64],
        names: &[String],
    ) -> Result<()> {
        self.observe_slot(market, seconds);
        if !adapter.requires_history() {
            return Ok(());
        }
        ensure!(
            (60..=seconds)
                .step_by(5)
                .all(|slot| self.observed.contains(&slot)),
            "UMR admission history unavailable until next complete market"
        );
        let p = adapter.directional_probability(features)?;
        let prior = self
            .values
            .iter()
            .filter(|v| v.0 < seconds)
            .collect::<Vec<_>>();
        for (name, lag) in [("probability_change_5s", 1), ("probability_change_15s", 3)] {
            if let Some(index) = names.iter().position(|v| v == name) {
                features[index] = prior
                    .len()
                    .checked_sub(lag)
                    .map(|i| p - prior[i].1)
                    .unwrap_or(0.0);
            }
        }
        if !self.values.iter().any(|v| v.0 == seconds) {
            self.values.push((seconds, p));
            self.values.sort_by_key(|v| v.0);
        }
        ensure!(
            self.values.len() <= 6,
            "UMR admission history exceeded frozen schedule"
        );
        Ok(())
    }
}
