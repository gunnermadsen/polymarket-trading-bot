import 'reflect-metadata';
import { WeatherDataSource } from './weather-datasource';

async function main(): Promise<void> {
  await WeatherDataSource.initialize();
  try {
    await WeatherDataSource.runMigrations({ transaction: 'all' });
  } finally {
    await WeatherDataSource.destroy();
  }
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
