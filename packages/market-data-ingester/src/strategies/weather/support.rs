use std::{process::Stdio, sync::Arc};

use chrono::{Datelike, TimeZone, Utc};
use tokio::{io::AsyncReadExt, process::Command};

use crate::domain::{
    BackfillContext, BackfillExecutionError, BackfillFailureKind, BackfillOutcome, BackfillRequest,
    BackfillShard, StrategyCapability, StrategyDescriptor, ValidatedBackfillRequest,
};

pub(super) struct WeatherBackfillSupport {
    descriptor: StrategyDescriptor,
}

impl WeatherBackfillSupport {
    pub(super) fn new(
        key: &'static str,
        name: &'static str,
        description: &'static str,
    ) -> Result<Self, BackfillExecutionError> {
        let descriptor = StrategyDescriptor {
            strategy_key: Arc::from(key),
            name: Arc::from(name),
            description: Arc::from(description),
            capabilities: vec![StrategyCapability::Backfill],
            strategy_contract_version: 1,
            request_schema_version: Some(1),
            shardable: true,
            maximum_shards: 1_200,
        };
        descriptor.validate()?;
        Ok(Self { descriptor })
    }

    pub(super) fn descriptor(&self) -> &StrategyDescriptor {
        &self.descriptor
    }

    pub(super) fn validate_request(
        &self,
        request: &BackfillRequest,
    ) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
        if request.strategy_key != self.descriptor.strategy_key.as_ref() {
            return Err(BackfillExecutionError::invalid(
                "strategy_key_mismatch",
                "request strategy key does not match the weather strategy",
            ));
        }
        if request.range.end <= request.range.start || request.range.end > Utc::now() {
            return Err(BackfillExecutionError::invalid(
                "range_invalid",
                "range must be increasing and may not end in the future",
            ));
        }
        if !request.parameters.is_object() {
            return Err(BackfillExecutionError::invalid(
                "parameters_invalid",
                "weather strategy parameters must be an object",
            ));
        }
        request.execution.validate()?;
        Ok(ValidatedBackfillRequest {
            strategy_key: self.descriptor.strategy_key.clone(),
            strategy_contract_version: 1,
            request_schema_version: 1,
            range_start: request.range.start,
            range_end: request.range.end,
            parameters: request.parameters.clone(),
            execution: request.execution.clone(),
        })
    }

    pub(super) fn plan_shards(
        &self,
        request: &ValidatedBackfillRequest,
    ) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
        let mut start = request.range_start;
        let mut shards = Vec::new();
        while start < request.range_end {
            let (year, month) = if start.month() == 12 {
                (start.year() + 1, 1)
            } else {
                (start.year(), start.month() + 1)
            };
            let next_month = Utc
                .with_ymd_and_hms(year, month, 1, 0, 0, 0)
                .single()
                .ok_or_else(|| {
                    BackfillExecutionError::invalid("range_invalid", "invalid month boundary")
                })?;
            let end = next_month.min(request.range_end);
            shards.push(BackfillShard {
                shard_key: format!("{}-{}", start.to_rfc3339(), end.to_rfc3339()),
                range_start: start,
                range_end: end,
                parameters: request.parameters.clone(),
            });
            if shards.len() > self.descriptor.maximum_shards {
                return Err(BackfillExecutionError::invalid(
                    "too_many_shards",
                    format!(
                        "request exceeds {} monthly shards",
                        self.descriptor.maximum_shards
                    ),
                ));
            }
            start = end;
        }
        Ok(shards)
    }

    pub(super) async fn execute_backfill(
        &self,
        context: BackfillContext,
        shard: BackfillShard,
    ) -> Result<BackfillOutcome, BackfillExecutionError> {
        let mut child = Command::new("nyc-temperature-model")
            .arg("execute-unified-backfill")
            .arg("--strategy-key")
            .arg(self.descriptor.strategy_key.as_ref())
            .arg("--job-id")
            .arg(context.job_id.to_string())
            .arg("--lease-token")
            .arg(context.lease_token.to_string())
            .arg("--worker-id")
            .arg(context.worker_id.as_ref())
            .arg("--start")
            .arg(shard.range_start.to_rfc3339())
            .arg("--end")
            .arg(shard.range_end.to_rfc3339())
            .arg("--parameters")
            .arg(shard.parameters.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| execution_error("weather_executor_spawn", error.to_string()))?;

        let mut stdout = child.stdout.take().ok_or_else(|| {
            execution_error(
                "weather_executor_stdout",
                "weather executor stdout unavailable",
            )
        })?;
        let mut stderr = child.stderr.take().ok_or_else(|| {
            execution_error(
                "weather_executor_stderr",
                "weather executor stderr unavailable",
            )
        })?;
        let stdout_task = tokio::spawn(async move {
            let mut bytes = Vec::new();
            stdout.read_to_end(&mut bytes).await.map(|_| bytes)
        });
        let stderr_task = tokio::spawn(async move {
            let mut bytes = Vec::new();
            stderr.read_to_end(&mut bytes).await.map(|_| bytes)
        });
        let status = tokio::select! {
            result = child.wait() => result.map_err(|error| execution_error("weather_executor_wait", error.to_string()))?,
            _ = context.shutdown.cancelled() => {
                let _ = child.kill().await;
                return Err(BackfillExecutionError::new(
                    BackfillFailureKind::Cancelled,
                    "worker_shutdown",
                    "worker shutdown interrupted weather backfill",
                ));
            }
        };
        let stdout = stdout_task
            .await
            .map_err(|error| execution_error("weather_executor_stdout", error.to_string()))?
            .map_err(|error| execution_error("weather_executor_stdout", error.to_string()))?;
        let stderr = stderr_task
            .await
            .map_err(|error| execution_error("weather_executor_stderr", error.to_string()))?
            .map_err(|error| execution_error("weather_executor_stderr", error.to_string()))?;
        if !status.success() {
            return Err(execution_error(
                "weather_executor_failed",
                String::from_utf8_lossy(&stderr).trim().to_owned(),
            ));
        }
        serde_json::from_slice(&stdout).map_err(|error| {
            execution_error(
                "weather_executor_output",
                format!("{error}: {}", String::from_utf8_lossy(&stdout)),
            )
        })
    }
}

fn execution_error(code: &'static str, message: impl Into<String>) -> BackfillExecutionError {
    BackfillExecutionError::new(BackfillFailureKind::TransientSource, code, message)
}
