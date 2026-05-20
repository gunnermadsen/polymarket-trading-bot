import 'reflect-metadata';
import { AppDataSource } from './datasource';

async function main(): Promise<void> {
  console.log('[db-migrate] Initializing data source');
  await AppDataSource.initialize();

  try {
    const hasPendingMigrations = await AppDataSource.showMigrations();
    console.log(`[db-migrate] Pending migrations: ${hasPendingMigrations ? 'yes' : 'no'}`);

    const appliedMigrations = await AppDataSource.runMigrations();
    console.log(`[db-migrate] Applied ${appliedMigrations.length} migration(s)`);
  } finally {
    await AppDataSource.destroy();
    console.log('[db-migrate] Data source closed');
  }
}

main().catch((error) => {
  const message = error instanceof Error ? error.stack ?? error.message : String(error);
  console.error('[db-migrate] Migration run failed');
  console.error(message);
  process.exitCode = 1;
});

