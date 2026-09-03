use chrono::{DateTime, Datelike, Timelike, Utc};
use reqwest::{Client, Url};
use serde_json::json;
use std::sync::Arc;

use crate::{
    domain::{
        BackfillExecutionError, BackfillRequest, BackfillShard, StrategyCapability,
        StrategyDescriptor, ValidatedBackfillRequest,
    },
    strategies::raw_archive::{self, RawObject},
};

pub struct Support {
    descriptor: StrategyDescriptor,
    pub client: Client,
}
impl Support {
    pub fn new(
        key: &'static str,
        name: &'static str,
        description: &'static str,
        maximum_shards: usize,
    ) -> Result<Self, BackfillExecutionError> {
        let descriptor = StrategyDescriptor {
            strategy_key: Arc::from(key),
            name: Arc::from(name),
            description: Arc::from(description),
            capabilities: vec![StrategyCapability::Backfill],
            strategy_contract_version: 1,
            request_schema_version: Some(1),
            shardable: true,
            maximum_shards,
        };
        descriptor.validate()?;
        Ok(Self {
            descriptor,
            client: raw_archive::client()?,
        })
    }
    pub fn descriptor(&self) -> &StrategyDescriptor {
        &self.descriptor
    }
    pub fn validate(
        &self,
        request: &BackfillRequest,
    ) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
        if request.strategy_key != self.descriptor.strategy_key.as_ref() {
            return Err(BackfillExecutionError::invalid(
                "strategy_key_mismatch",
                "request strategy key did not match raw weather strategy",
            ));
        }
        if request.range.end <= request.range.start || request.range.end > Utc::now() {
            return Err(BackfillExecutionError::invalid(
                "range_invalid",
                "range must be increasing and may not end in the future",
            ));
        }
        if !request.parameters.as_object().is_some_and(|v| v.is_empty()) {
            return Err(BackfillExecutionError::invalid(
                "parameters_invalid",
                "raw weather strategies accept no parameters",
            ));
        }
        request.execution.validate()?;
        Ok(ValidatedBackfillRequest {
            strategy_key: self.descriptor.strategy_key.clone(),
            strategy_contract_version: 1,
            request_schema_version: 1,
            range_start: request.range.start,
            range_end: request.range.end,
            parameters: json!({}),
            execution: request.execution.clone(),
        })
    }
    pub fn hourly(
        &self,
        request: &ValidatedBackfillRequest,
    ) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
        shards(
            request,
            self.descriptor.maximum_shards,
            chrono::Duration::hours(1),
        )
    }
    pub fn daily(
        &self,
        request: &ValidatedBackfillRequest,
    ) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
        shards(
            request,
            self.descriptor.maximum_shards,
            chrono::Duration::days(1),
        )
    }
}
fn shards(
    request: &ValidatedBackfillRequest,
    maximum: usize,
    width: chrono::Duration,
) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
    let mut cursor = request.range_start;
    let mut out = Vec::new();
    while cursor < request.range_end {
        let end = (cursor + width).min(request.range_end);
        out.push(BackfillShard {
            shard_key: format!(
                "{}-{}",
                cursor.format("%Y%m%dT%H%M%SZ"),
                end.format("%Y%m%dT%H%M%SZ")
            ),
            range_start: cursor,
            range_end: end,
            parameters: json!({}),
        });
        if out.len() > maximum {
            return Err(BackfillExecutionError::invalid(
                "too_many_shards",
                "request exceeds strategy shard limit",
            ));
        }
        cursor = end;
    }
    Ok(out)
}

pub async fn asos_objects(
    client: &Client,
    shard: &BackfillShard,
    one_minute: bool,
) -> Result<Vec<RawObject>, BackfillExecutionError> {
    let base = if one_minute {
        std::env::var("IEM_ASOS_ONE_MINUTE_URL").unwrap_or_else(|_| {
            "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py".into()
        })
    } else {
        std::env::var("IEM_ASOS_METAR_URL")
            .unwrap_or_else(|_| "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py".into())
    };
    let mut url = Url::parse(&base).map_err(invalid)?;
    {
        let mut q = url.query_pairs_mut();
        if one_minute {
            q.append_pair("station", "LGA")
                .append_pair("vars", "tmpf")
                .append_pair(
                    "sts",
                    &shard.range_start.format("%Y-%m-%dT%H:%MZ").to_string(),
                )
                .append_pair(
                    "ets",
                    &shard.range_end.format("%Y-%m-%dT%H:%MZ").to_string(),
                )
                .append_pair("sample", "1min")
                .append_pair("what", "download")
                .append_pair("tz", "UTC")
                .append_pair("delim", "comma")
                .append_pair("gis", "no");
        } else {
            for (k, v) in [
                ("station", "LGA"),
                ("data", "tmpf"),
                ("year1", &shard.range_start.year().to_string()),
                ("month1", &shard.range_start.month().to_string()),
                ("day1", &shard.range_start.day().to_string()),
                ("year2", &shard.range_end.year().to_string()),
                ("month2", &shard.range_end.month().to_string()),
                ("day2", &shard.range_end.day().to_string()),
                ("tz", "Etc/UTC"),
                ("format", "onlycomma"),
                ("latlon", "no"),
                ("elev", "no"),
                ("missing", "empty"),
                ("trace", "empty"),
                ("direct", "no"),
            ] {
                q.append_pair(k, v);
            }
            q.append_pair("report_type", "3")
                .append_pair("report_type", "4");
        }
    }
    let _ = client;
    let kind = if one_minute { "one-minute" } else { "metar" };
    let stamp = shard.range_start.format("%Y%m%dT%H%M%SZ");
    Ok(vec![RawObject {
        logical_key: format!(
            "iem:asos:{kind}:{stamp}:{}",
            shard.range_end.format("%Y%m%dT%H%M%SZ")
        ),
        provider: if one_minute {
            "iem_ncei_asos_one_minute"
        } else {
            "iem_asos_metar"
        },
        source_uri: url.to_string(),
        relative_path: format!(
            "iem-asos/{kind}/{}/{:02}/{:02}/{stamp}.csv",
            shard.range_start.year(),
            shard.range_start.month(),
            shard.range_start.day()
        )
        .into(),
        media_type: "text/csv",
        minimum: shard.range_start,
        maximum: shard.range_end,
    }])
}

pub async fn hrrr_objects(shard: &BackfillShard) -> Result<Vec<RawObject>, BackfillExecutionError> {
    let valid = shard.range_start;
    let available = valid - chrono::Duration::minutes(75);
    let run_hour = available.hour() - (available.hour() % 6);
    let run = available
        .with_hour(run_hour)
        .and_then(|v| v.with_minute(0))
        .and_then(|v| v.with_second(0))
        .ok_or_else(|| invalid("invalid HRRR cycle"))?;
    let lead = (valid - run).num_hours().max(0);
    let date = run.format("%Y%m%d");
    let name = format!("hrrr.t{:02}z.wrfsfcf{:02}.grib2", run.hour(), lead);
    let uri = format!("https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date}/conus/{name}");
    Ok(vec![RawObject {
        logical_key: format!("noaa:hrrr:sfc:{date}T{:02}:f{lead:02}", run.hour()),
        provider: "noaa_hrrr_open_data",
        source_uri: uri,
        relative_path: format!("noaa-hrrr/{date}/t{:02}/f{lead:02}/{name}", run.hour()).into(),
        media_type: "application/x-grib2",
        minimum: valid,
        maximum: shard.range_end,
    }])
}

pub async fn goes_objects(
    client: &Client,
    shard: &BackfillShard,
) -> Result<Vec<RawObject>, BackfillExecutionError> {
    let satellite = if shard.range_start
        < DateTime::parse_from_rfc3339("2025-04-07T15:10:00Z")
            .unwrap()
            .with_timezone(&Utc)
    {
        "16"
    } else {
        "19"
    };
    let products = [
        ("ABI-L2-CMIPC", Some("C13")),
        ("ABI-L2-ACMC", None),
        ("ABI-L2-ACHTF", None),
        ("ABI-L2-ACHAC", None),
        ("ABI-L2-CMIPC", Some("C02")),
        ("ABI-L2-CODC", None),
        ("ABI-L2-CMIPC", Some("C08")),
    ];
    let mut out = Vec::new();
    for (product, channel) in products {
        let prefix = format!(
            "{product}/{}/{:03}/{:02}/",
            shard.range_start.year(),
            shard.range_start.ordinal(),
            shard.range_start.hour()
        );
        let url =
            format!("https://noaa-goes{satellite}.s3.amazonaws.com/?list-type=2&prefix={prefix}");
        let text = client
            .get(&url)
            .send()
            .await
            .map_err(source)?
            .error_for_status()
            .map_err(source)?
            .text()
            .await
            .map_err(source)?;
        for key in xml_keys(&text)
            .into_iter()
            .filter(|k| channel.is_none_or(|c| k.contains(c)))
        {
            let file = key
                .rsplit('/')
                .next()
                .ok_or_else(|| invalid("GOES object key lacked filename"))?;
            out.push(RawObject {
                logical_key: format!("noaa:goes{satellite}:{key}"),
                provider: "noaa_goes_open_data",
                source_uri: format!("https://noaa-goes{satellite}.s3.amazonaws.com/{key}"),
                relative_path: format!(
                    "noaa-goes/abi/G{satellite}/{product}/{}/{:03}/{:02}/{file}",
                    shard.range_start.year(),
                    shard.range_start.ordinal(),
                    shard.range_start.hour()
                )
                .into(),
                media_type: "application/x-netcdf",
                minimum: shard.range_start,
                maximum: shard.range_end,
            });
        }
    }
    if out.is_empty() {
        return Err(source("GOES catalog returned no source objects"));
    }
    Ok(out)
}
fn xml_keys(text: &str) -> Vec<String> {
    text.split("<Key>")
        .skip(1)
        .filter_map(|v| v.split("</Key>").next())
        .map(|v| v.replace("&amp;", "&"))
        .collect()
}
fn source(error: impl std::fmt::Display) -> BackfillExecutionError {
    BackfillExecutionError::new(
        crate::domain::BackfillFailureKind::TransientSource,
        "weather_source",
        error.to_string(),
    )
}
fn invalid(error: impl std::fmt::Display) -> BackfillExecutionError {
    BackfillExecutionError::invalid("weather_source_invalid", error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_s3_keys() {
        assert_eq!(
            xml_keys("<Key>a.nc</Key><Key>b.nc</Key>"),
            vec!["a.nc", "b.nc"]
        );
    }
}
