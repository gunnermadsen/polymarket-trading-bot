import 'dotenv/config';
import { join } from 'path';
import { DataSource } from 'typeorm';
import { getPostgresSslConfig } from './postgres-ssl.config';

export const AppDataSource = new DataSource({
  type: 'postgres',
  host: process.env.POSTGRES_HOST || 'timescaledb',
  port: +(process.env.POSTGRES_PORT || 5432),
  username: process.env.POSTGRES_USER || 'postgres',
  password: process.env.POSTGRES_PASSWORD || 'postgres',
  database: process.env.POSTGRES_DB || 'capitonic_timescale',
  ssl: getPostgresSslConfig(),
  synchronize: false,
  entities: [],
  migrations: [join(__dirname, 'migrations/*.{ts,js}')],
  extra: {
    application_name: 'capitonic-db-migrate',
  },
});
