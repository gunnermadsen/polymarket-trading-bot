#!/usr/bin/env node

import {
  createClient,
  decodeReport,
  getReportVersion,
} from '@chainlink/data-streams-sdk';

const DEFAULT_ENDPOINT = 'https://api.dataengine.chain.link';
const DEFAULT_WS_ENDPOINT = 'wss://ws.dataengine.chain.link';
const DEFAULT_START = '2026-08-01T00:00:00Z';
const DEFAULT_LIMIT = 5;

function usage(message) {
  if (message) console.error(message);
  console.error(`usage: node scripts/test-chainlink-twap-history.mjs [options]

Read-only Chainlink Data Streams TWAP historical availability probe.

Options:
  --start <UTC ISO>       Historical query start (default: ${DEFAULT_START})
  --limit <1-10000>       Reports requested per matching feed (default: ${DEFAULT_LIMIT})
  --window <30|60|all>    Limit automatic feed discovery (default: all)
  --feed-id <id[,id...]>  Query exact feed IDs instead of discovered BTC/USD TWAP feeds
  --list-only             List matching feeds without querying reports
  --include-full-report   Include signed fullReport blobs in output
  --help                  Show this message

Credentials are read from:
  POLYMARKET_CHAINLINK_DATA_STREAMS_API_KEY
  POLYMARKET_CHAINLINK_DATA_STREAMS_API_SECRET

Optional endpoints:
  POLYMARKET_CHAINLINK_DATA_STREAMS_REST_URL
  POLYMARKET_CHAINLINK_DATA_STREAMS_WS_URL`);
  process.exit(message ? 2 : 0);
}

function parseArguments(argv) {
  const options = {
    start: DEFAULT_START,
    limit: DEFAULT_LIMIT,
    window: 'all',
    feedIds: [],
    listOnly: false,
    includeFullReport: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--help') usage();
    if (argument === '--list-only') {
      options.listOnly = true;
      continue;
    }
    if (argument === '--include-full-report') {
      options.includeFullReport = true;
      continue;
    }
    if (!['--start', '--limit', '--window', '--feed-id'].includes(argument)) {
      usage(`unexpected argument: ${argument}`);
    }
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) usage(`missing value for ${argument}`);
    index += 1;
    if (argument === '--start') options.start = value;
    if (argument === '--limit') options.limit = Number(value);
    if (argument === '--window') options.window = value;
    if (argument === '--feed-id') {
      options.feedIds.push(...value.split(',').map((item) => item.trim()).filter(Boolean));
    }
  }
  if (!['30', '60', 'all'].includes(options.window)) {
    usage('--window must be 30, 60, or all');
  }
  if (!Number.isInteger(options.limit) || options.limit < 1 || options.limit > 10_000) {
    usage('--limit must be an integer between 1 and 10000');
  }
  const startMilliseconds = Date.parse(options.start);
  if (!Number.isFinite(startMilliseconds) || !options.start.endsWith('Z')) {
    usage('--start must be an ISO-8601 UTC timestamp ending in Z');
  }
  options.startTimestamp = Math.floor(startMilliseconds / 1_000);
  for (const feedId of options.feedIds) {
    if (!/^0x[0-9a-fA-F]{64}$/.test(feedId)) usage(`invalid feed ID: ${feedId}`);
  }
  options.feedIds = [...new Set(options.feedIds.map((feedId) => feedId.toLowerCase()))];
  return options;
}

function requiredEnvironment(name) {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function detectsWindow(feed, seconds) {
  const text = `${feed.name ?? ''} ${feed.asset ?? ''}/${feed.quoteAsset ?? ''}`.toLowerCase();
  const token = seconds === 30 ? '(?:30|thirty)' : '(?:60|sixty)';
  return new RegExp(`${token}\\s*(?:s|sec|secs|second|seconds)`).test(text)
    || new RegExp(`twap[^0-9a-z]*${token}`).test(text)
    || new RegExp(`${token}[^0-9a-z]*twap`).test(text);
}

function isBtcUsdTwap(feed, window) {
  const asset = String(feed.asset ?? '').toUpperCase();
  const quote = String(feed.quoteAsset ?? '').toUpperCase();
  const name = String(feed.name ?? '').toLowerCase();
  const pairMatches = (asset === 'BTC' && quote === 'USD')
    || (/btc/.test(name) && /usd/.test(name));
  if (!pairMatches || !name.includes('twap')) return false;
  if (window === 'all') return true;
  return detectsWindow(feed, Number(window));
}

function jsonValue(value) {
  if (typeof value === 'bigint') return value.toString();
  if (Array.isArray(value)) return value.map(jsonValue);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, jsonValue(item)]));
  }
  return value;
}

function isoTimestamp(seconds) {
  return Number.isFinite(seconds) ? new Date(seconds * 1_000).toISOString() : null;
}

function scaledDecimal(value, decimals) {
  if (typeof value !== 'bigint' || !Number.isInteger(decimals) || decimals < 0) return null;
  const negative = value < 0n;
  const digits = (negative ? -value : value).toString().padStart(decimals + 1, '0');
  const split = digits.length - decimals;
  const result = decimals === 0 ? digits : `${digits.slice(0, split)}.${digits.slice(split)}`;
  return negative ? `-${result}` : result;
}

function reportOutput(report, feed, includeFullReport) {
  const base = {
    feedID: report.feedID,
    reportVersion: getReportVersion(report.feedID),
    observationsTimestamp: report.observationsTimestamp,
    observationsTime: isoTimestamp(report.observationsTimestamp),
    validFromTimestamp: report.validFromTimestamp,
    validFromTime: isoTimestamp(report.validFromTimestamp),
  };
  if (includeFullReport) base.fullReport = report.fullReport;
  try {
    const decoded = decodeReport(report.fullReport, report.feedID);
    const price = decoded.price ?? decoded.midPrice;
    return {
      ...base,
      decoded: jsonValue(decoded),
      scaledPrice: scaledDecimal(price, feed?.decimals),
    };
  } catch (error) {
    return { ...base, decodeError: error instanceof Error ? error.message : String(error) };
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const endpoint = process.env.POLYMARKET_CHAINLINK_DATA_STREAMS_REST_URL?.trim()
    || DEFAULT_ENDPOINT;
  const wsEndpoint = process.env.POLYMARKET_CHAINLINK_DATA_STREAMS_WS_URL?.trim()
    || DEFAULT_WS_ENDPOINT;
  const client = createClient({
    apiKey: requiredEnvironment('POLYMARKET_CHAINLINK_DATA_STREAMS_API_KEY'),
    userSecret: requiredEnvironment('POLYMARKET_CHAINLINK_DATA_STREAMS_API_SECRET'),
    endpoint,
    wsEndpoint,
    retryAttempts: 1,
    timeout: 30_000,
  });

  const availableFeeds = await client.listFeeds();
  const feedById = new Map(availableFeeds.map((feed) => [feed.feedID.toLowerCase(), feed]));
  const selectedFeeds = options.feedIds.length > 0
    ? options.feedIds.map((feedID) => feedById.get(feedID) ?? {
      feedID,
      name: 'explicit feed ID (not returned by listFeeds)',
      decimals: null,
      asset: null,
      quoteAsset: null,
    })
    : availableFeeds.filter((feed) => isBtcUsdTwap(feed, options.window));

  const output = {
    probe: 'chainlink_data_streams_historical_twap',
    sdkVersion: '1.2.1',
    endpoint,
    startTimestamp: options.startTimestamp,
    startTime: isoTimestamp(options.startTimestamp),
    requestedLimit: options.limit,
    entitledFeedCount: availableFeeds.length,
    ...(options.listOnly ? { entitledFeeds: availableFeeds.map(jsonValue) } : {}),
    matchingFeeds: selectedFeeds.map(jsonValue),
    results: [],
  };

  if (selectedFeeds.length === 0) {
    output.warning = 'No entitled BTC/USD TWAP feeds matched. Inspect listFeeds output or pass --feed-id.';
  } else if (!options.listOnly) {
    for (const feed of selectedFeeds) {
      try {
        const reports = await client.getReportsPage(
          feed.feedID,
          options.startTimestamp,
          options.limit,
        );
        output.results.push({
          feed: jsonValue(feed),
          reportCount: reports.length,
          earliestObservationTime: reports.length
            ? isoTimestamp(reports[0].observationsTimestamp)
            : null,
          latestObservationTime: reports.length
            ? isoTimestamp(reports.at(-1).observationsTimestamp)
            : null,
          reports: reports.map((report) => reportOutput(
            report,
            feed,
            options.includeFullReport,
          )),
        });
      } catch (error) {
        output.results.push({
          feed: jsonValue(feed),
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
  }

  console.log(JSON.stringify(output, null, 2));
  if (selectedFeeds.length === 0 || output.results.some((result) => result.error)) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(JSON.stringify({
    probe: 'chainlink_data_streams_historical_twap',
    fatalError: error instanceof Error ? error.message : String(error),
  }, null, 2));
  process.exitCode = 1;
});
