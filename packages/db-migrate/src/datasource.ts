import { config } from 'dotenv';
import { join } from 'path';
import { DataSource } from 'typeorm';
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

export const AppDataSource = new DataSource({
  type: 'postgres',
  host: process.env.POSTGRES_HOST || 'timescaledb-0',
  port: +(process.env.POSTGRES_PORT || 5432),
  username: process.env.POSTGRES_USER || 'postgres',
  password: requiredEnv('POSTGRES_PASSWORD'),
  database: process.env.POSTGRES_DB || 'polymarket',
  ssl: getPostgresSslConfig(),
  synchronize: false,
  entities: [],
  migrations: [join(__dirname, 'migrations/*.{ts,js}')],
  extra: {
    application_name: 'polymarket-db-migrate',
  },
});
