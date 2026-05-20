import { readFileSync } from 'fs';

type PostgresSslMode = 'disable' | 'require' | 'verify-ca' | 'verify-full';

function getSslMode(): PostgresSslMode {
  const rawMode = (process.env.POSTGRES_SSL_MODE || 'disable').toLowerCase();
  if (rawMode === 'require' || rawMode === 'verify-ca' || rawMode === 'verify-full') {
    return rawMode;
  }
  return 'disable';
}

function readOptional(filePath: string | undefined): string | undefined {
  if (!filePath) {
    return undefined;
  }
  return readFileSync(filePath, 'utf8');
}

export function getPostgresSslConfig() {
  const mode = getSslMode();
  if (mode === 'disable') {
    return undefined;
  }

  const host = process.env.POSTGRES_SSL_SERVERNAME || process.env.POSTGRES_HOST || 'timescaledb';
  const caFile = process.env.POSTGRES_SSL_CA_FILE;
  if (!caFile) {
    throw new Error('POSTGRES_SSL_CA_FILE is required when POSTGRES_SSL_MODE is enabled.');
  }

  const certFile = process.env.POSTGRES_SSL_CERT_FILE;
  const keyFile = process.env.POSTGRES_SSL_KEY_FILE;
  if ((certFile && !keyFile) || (!certFile && keyFile)) {
    throw new Error('POSTGRES_SSL_CERT_FILE and POSTGRES_SSL_KEY_FILE must be set together.');
  }

  const ssl: Record<string, unknown> = {
    ca: readFileSync(caFile, 'utf8'),
    servername: host,
    rejectUnauthorized: mode !== 'require',
  };

  const cert = readOptional(certFile);
  const key = readOptional(keyFile);
  if (cert && key) {
    ssl.cert = cert;
    ssl.key = key;
  }

  if (mode === 'verify-ca') {
    ssl.checkServerIdentity = () => undefined;
  }

  return ssl;
}

