import { MigrationInterface, QueryRunner } from 'typeorm';

const serviceRoles = [
  ['capitonic_trading', 'CAPITONIC_TRADING_POSTGRES_PASSWORD'],
  ['capitonic_ingester_master', 'CAPITONIC_INGESTER_MASTER_POSTGRES_PASSWORD'],
  ['capitonic_ingester_worker', 'CAPITONIC_INGESTER_WORKER_POSTGRES_PASSWORD'],
  ['capitonic_grafana', 'CAPITONIC_GRAFANA_POSTGRES_PASSWORD'],
] as const;

function requiredPassword(environmentVariable: string): string {
  const password = process.env[environmentVariable];
  if (!password || password.trim() === '') {
    throw new Error(`${environmentVariable} is required to provision PostgreSQL service roles.`);
  }
  if (password.includes('\0')) {
    throw new Error(`${environmentVariable} must not contain a null byte.`);
  }
  return password;
}

function quoteLiteral(value: string): string {
  return `'${value.replaceAll("'", "''")}'`;
}

export class AddPostgresServiceIdentities1789131600000
  implements MigrationInterface
{
  name = 'AddPostgresServiceIdentities1789131600000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    for (const [role, environmentVariable] of serviceRoles) {
      await queryRunner.query(`
        DO $role$
        BEGIN
          IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${role}') THEN
            CREATE ROLE ${role} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT;
          END IF;
        END
        $role$;

        ALTER ROLE ${role}
          LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT
          PASSWORD ${quoteLiteral(requiredPassword(environmentVariable))};
      `);
    }

    await queryRunner.query(`
      GRANT CONNECT ON DATABASE polymarket TO
        capitonic_trading,
        capitonic_ingester_master,
        capitonic_ingester_worker,
        capitonic_grafana;

      GRANT USAGE ON SCHEMA polymarket, market_data TO capitonic_trading;
      GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA polymarket TO capitonic_trading;
      GRANT SELECT ON ALL TABLES IN SCHEMA market_data TO capitonic_trading;
      GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA polymarket TO capitonic_trading;

      GRANT USAGE ON SCHEMA ingester TO capitonic_ingester_master;
      GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA ingester TO capitonic_ingester_master;
      GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA ingester TO capitonic_ingester_master;

      GRANT USAGE ON SCHEMA ingester, market_data, polymarket TO capitonic_ingester_worker;
      GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA ingester, market_data, polymarket TO capitonic_ingester_worker;
      GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA ingester, market_data, polymarket TO capitonic_ingester_worker;

      GRANT USAGE ON SCHEMA ingester, market_data, polymarket TO capitonic_grafana;
      GRANT SELECT ON ALL TABLES IN SCHEMA ingester, market_data, polymarket TO capitonic_grafana;
      ALTER ROLE capitonic_grafana SET default_transaction_read_only = on;

      ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA polymarket
        GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO capitonic_trading;
      ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA market_data
        GRANT SELECT ON TABLES TO capitonic_trading;
      ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA polymarket
        GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO capitonic_trading;

      ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA ingester
        GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO capitonic_ingester_master;
      ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA ingester
        GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO capitonic_ingester_master;

      ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA ingester, market_data, polymarket
        GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO capitonic_ingester_worker;
      ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA ingester, market_data, polymarket
        GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO capitonic_ingester_worker;

      ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA ingester, market_data, polymarket
        GRANT SELECT ON TABLES TO capitonic_grafana;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA ingester, market_data, polymarket
        REVOKE SELECT ON TABLES FROM capitonic_grafana;

      ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA ingester, market_data, polymarket
        REVOKE USAGE, SELECT, UPDATE ON SEQUENCES FROM capitonic_ingester_worker;
      ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA ingester, market_data, polymarket
        REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM capitonic_ingester_worker;

      ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA ingester
        REVOKE USAGE, SELECT, UPDATE ON SEQUENCES FROM capitonic_ingester_master;
      ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA ingester
        REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM capitonic_ingester_master;

      ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA polymarket
        REVOKE USAGE, SELECT, UPDATE ON SEQUENCES FROM capitonic_trading;
      ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA market_data
        REVOKE SELECT ON TABLES FROM capitonic_trading;
      ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA polymarket
        REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM capitonic_trading;

      REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA ingester, market_data, polymarket FROM capitonic_grafana;
      REVOKE USAGE ON SCHEMA ingester, market_data, polymarket FROM capitonic_grafana;

      REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ingester, market_data, polymarket FROM capitonic_ingester_worker;
      REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA ingester, market_data, polymarket FROM capitonic_ingester_worker;
      REVOKE USAGE ON SCHEMA ingester, market_data, polymarket FROM capitonic_ingester_worker;

      REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ingester FROM capitonic_ingester_master;
      REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA ingester FROM capitonic_ingester_master;
      REVOKE USAGE ON SCHEMA ingester FROM capitonic_ingester_master;

      REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA polymarket FROM capitonic_trading;
      REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA polymarket, market_data FROM capitonic_trading;
      REVOKE USAGE ON SCHEMA polymarket, market_data FROM capitonic_trading;

      REVOKE CONNECT ON DATABASE polymarket FROM
        capitonic_trading,
        capitonic_ingester_master,
        capitonic_ingester_worker,
        capitonic_grafana;

      DROP ROLE capitonic_grafana;
      DROP ROLE capitonic_ingester_worker;
      DROP ROLE capitonic_ingester_master;
      DROP ROLE capitonic_trading;
    `);
  }
}
