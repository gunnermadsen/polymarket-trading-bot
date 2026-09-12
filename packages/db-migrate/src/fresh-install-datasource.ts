import { config } from 'dotenv';
import { DataSource } from 'typeorm';
import { EstablishCurrentSchemaBaseline1789200000000 } from './fresh-install/1789200000000-EstablishCurrentSchemaBaseline';
import { getPostgresSslConfig } from './postgres-ssl.config';

config({ path: '.env' });
config({ path: '.env.postgres', override: true });

function requiredEnv(key: string): string {
  const value = process.env[key];
  if (!value || value.trim() === '') {
    throw new Error(`${key} is required.`);
  }
  return value;
}

export const FreshInstallDataSource = new DataSource({
  type: 'postgres',
  host: process.env.POSTGRES_HOST || 'timescaledb-0',
  port: +(process.env.POSTGRES_PORT || 5432),
  username: process.env.POSTGRES_USER || 'postgres',
  password: requiredEnv('POSTGRES_PASSWORD'),
  database: process.env.POSTGRES_DB || 'polymarket',
  ssl: getPostgresSslConfig(),
  synchronize: false,
  entities: [],
  migrations: [EstablishCurrentSchemaBaseline1789200000000],
  extra: {
    application_name: 'polymarket-db-migrate-fresh-install',
  },
});
