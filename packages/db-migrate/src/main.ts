import 'reflect-metadata';
import { AppDataSource } from './datasource';
import { FreshInstallDataSource } from './fresh-install-datasource';

async function establishFreshInstallBaseline(): Promise<void> {
  console.log('[db-migrate] Checking for an existing migration ledger');
  await AppDataSource.initialize();

  let hasMigrationLedger: boolean;
  try {
    const [{ migration_ledger: migrationLedger }] = (await AppDataSource.query(
      `SELECT to_regclass('public.migrations') IS NOT NULL AS migration_ledger`,
    )) as Array<{ migration_ledger: boolean }>;
    hasMigrationLedger = migrationLedger;
  } finally {
    await AppDataSource.destroy();
  }

  if (hasMigrationLedger) {
    console.log('[db-migrate] Existing migration ledger found; skipping baseline');
    return;
  }

  console.log('[db-migrate] Empty database detected; establishing current schema baseline');
  await FreshInstallDataSource.initialize();
  try {
    const appliedMigrations = await FreshInstallDataSource.runMigrations({
      transaction: 'each',
    });
    if (appliedMigrations.length !== 1) {
      throw new Error(
        `Expected one fresh-install baseline migration, applied ${appliedMigrations.length}.`,
      );
    }
    console.log('[db-migrate] Current schema baseline established');
  } finally {
    await FreshInstallDataSource.destroy();
  }
}

async function main(): Promise<void> {
  await establishFreshInstallBaseline();
  console.log('[db-migrate] Initializing data source');
  await AppDataSource.initialize();

  try {
    const hasPendingMigrations = await AppDataSource.showMigrations();
    console.log(`[db-migrate] Pending migrations: ${hasPendingMigrations ? 'yes' : 'no'}`);

    const appliedMigrations = await AppDataSource.runMigrations({ transaction: 'each' });
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
