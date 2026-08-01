import { config } from 'dotenv';
import { join } from 'path';
import { DataSource } from 'typeorm';

config({ path: '.env.weather' });

export const WeatherDataSource = new DataSource({
  type: 'postgres',
  host: process.env.POSTGRES_HOST || 'temperature-postgres',
  port: +(process.env.POSTGRES_PORT || 5432),
  username: process.env.POSTGRES_USER || 'postgres',
  password: process.env.POSTGRES_PASSWORD || undefined,
  database: process.env.POSTGRES_DB || 'temperature_expectancy',
  ssl: false,
  synchronize: false,
  entities: [],
  migrations: [join(__dirname, 'migrations/weather/*.{ts,js}')],
  migrationsTableName: 'weather_migrations',
  extra: {
    application_name: 'temperature-expectancy-db-migrate',
  },
});
