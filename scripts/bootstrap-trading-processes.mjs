#!/usr/bin/env node

import { readdirSync, readFileSync } from "node:fs";
import { basename, extname, join } from "node:path";

const DEFAULT_BASE_URL = "http://127.0.0.1:8097";
const DEFAULT_ENDPOINT = "/admin/trading-processes/by-key";
const DEFAULT_TEMPLATE_DIR = "infra/processes";

function parseArgs(argv) {
  const options = {
    baseUrl: process.env.POLYMARKET_HTTP_BASE_URL || process.env.POLYMARKET_ADMIN_BASE_URL || DEFAULT_BASE_URL,
    endpoint: process.env.POLYMARKET_PROCESS_UPSERT_ENDPOINT || DEFAULT_ENDPOINT,
    token: process.env.POLYMARKET_HTTP_ADMIN_TOKEN || "",
    templateDir: DEFAULT_TEMPLATE_DIR,
    files: [],
    dryRun: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--base-url") {
      options.baseUrl = requiredValue(argv, ++i, arg);
    } else if (arg === "--endpoint") {
      options.endpoint = requiredValue(argv, ++i, arg);
    } else if (arg === "--token") {
      options.token = requiredValue(argv, ++i, arg);
    } else if (arg === "--template-dir") {
      options.templateDir = requiredValue(argv, ++i, arg);
    } else if (arg === "--file") {
      options.files.push(requiredValue(argv, ++i, arg));
    } else if (arg === "--dry-run") {
      options.dryRun = true;
    } else if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

function requiredValue(argv, index, arg) {
  const value = argv[index];
  if (!value || value.startsWith("--")) {
    throw new Error(`Missing value for ${arg}`);
  }
  return value;
}

function printHelp() {
  console.log(`Usage:
  node scripts/bootstrap-trading-processes.mjs [--template-dir infra/processes] [--file path/to/process.json] [--base-url http://127.0.0.1:8097] [--endpoint /admin/trading-processes/by-key] [--token token] [--dry-run]

Reads managed BTC realtime-paper process JSON templates and upserts them by process key through the local admin API.
POLYMARKET_HTTP_ADMIN_TOKEN, POLYMARKET_HTTP_BASE_URL, POLYMARKET_ADMIN_BASE_URL, and POLYMARKET_PROCESS_UPSERT_ENDPOINT are also supported.
`);
}

function templatePaths(options) {
  if (options.files.length > 0) {
    return options.files;
  }

  return readdirSync(options.templateDir)
    .filter((file) => extname(file) === ".json")
    .sort()
    .map((file) => join(options.templateDir, file));
}

function readTemplate(pathname) {
  const template = JSON.parse(readFileSync(pathname, "utf8"));
  const required = ["name", "process_type", "process_scope", "process_key", "enabled", "status"];
  const missing = required.filter((key) => template[key] == null || template[key] === "");
  if (missing.length > 0) {
    throw new Error(`${pathname} is missing required keys: ${missing.join(", ")}`);
  }
  if (template.process_type !== "btc_5m" || template.process_scope !== "realtime_paper") {
    throw new Error(`${pathname} must define a btc_5m/realtime_paper process`);
  }
  if (template.enabled !== false) {
    throw new Error(`${pathname} must be inactive; start it through the process /start endpoint`);
  }
  if (!["created", "stopped", "failed", "completed"].includes(template.status)) {
    throw new Error(`${pathname} has an unsupported inactive status: ${template.status}`);
  }
  return template;
}

function apiUrl(baseUrl, endpoint, processKey) {
  const trimmedBase = baseUrl.replace(/\/+$/, "");
  const normalizedEndpoint = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  return `${trimmedBase}${normalizedEndpoint}/${encodeURIComponent(processKey)}`;
}

async function upsertTemplate(baseUrl, endpoint, token, template) {
  const url = apiUrl(baseUrl, endpoint, template.process_key);
  const response = await fetch(url, {
    method: "PUT",
    headers: {
      "authorization": `Bearer ${token}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(template),
  });

  const text = await response.text();
  let body;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }

  if (!response.ok) {
    const detail = typeof body === "string" ? body : JSON.stringify(body);
    throw new Error(`Upsert failed for ${template.process_key}: ${response.status} ${detail}`);
  }

  return body;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const paths = templatePaths(options);
  const templates = paths.map((pathname) => ({
    pathname,
    template: readTemplate(pathname),
  }));

  if (templates.length === 0) {
    throw new Error("No process templates found");
  }

  console.log(`Loaded ${templates.length} trading process templates`);
  console.log(`Upsert endpoint: ${apiUrl(options.baseUrl, options.endpoint, "{process_key}")}`);

  if (options.dryRun) {
    for (const { pathname, template } of templates) {
      console.log(`Dry run: ${basename(pathname)} -> ${template.process_key}`);
    }
    return;
  }

  if (!options.token) {
    throw new Error("POLYMARKET_HTTP_ADMIN_TOKEN or --token is required");
  }

  for (const { pathname, template } of templates) {
    await upsertTemplate(options.baseUrl, options.endpoint, options.token, template);
    console.log(`Upserted ${basename(pathname)} -> ${template.process_key}`);
  }
}

try {
  await main();
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
