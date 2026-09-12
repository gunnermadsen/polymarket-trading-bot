import { MigrationInterface, QueryRunner } from 'typeorm';

const STRATEGY_KEY = 'binance_spot_btcusdt_one_second_ohlcv';
const PREVIOUS_WEBSOCKET_URL =
  'wss://stream.binance.com:9443/ws/btcusdt@kline_1s';
const STANDARD_WEBSOCKET_URL =
  'wss://stream.binance.com:443/ws/btcusdt@kline_1s';

export class UseStandardBinanceOneSecondWebsocketPort1789372800000
  implements MigrationInterface
{
  name = 'UseStandardBinanceOneSecondWebsocketPort1789372800000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await this.replaceWebsocketUrl(
      queryRunner,
      PREVIOUS_WEBSOCKET_URL,
      STANDARD_WEBSOCKET_URL,
    );
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await this.replaceWebsocketUrl(
      queryRunner,
      STANDARD_WEBSOCKET_URL,
      PREVIOUS_WEBSOCKET_URL,
    );
  }

  private async replaceWebsocketUrl(
    queryRunner: QueryRunner,
    expectedUrl: string,
    replacementUrl: string,
  ): Promise<void> {
    const result = await queryRunner.query(
      `
        UPDATE ingester.profiles
        SET config = jsonb_set(config, '{websocket_url}', to_jsonb($1::text), false),
            desired_generation = desired_generation + 1,
            updated_at = NOW()
        WHERE strategy_key = $2
          AND config_schema_version = 1
          AND config->>'websocket_url' = $3
        RETURNING strategy_key
      `,
      [replacementUrl, STRATEGY_KEY, expectedUrl],
    );

    const rows = Array.isArray(result) && Array.isArray(result[0]) ? result[0] : result;
    if (!Array.isArray(rows) || rows.length !== 1) {
      throw new Error(
        `Expected exactly one ${STRATEGY_KEY} profile using ${expectedUrl}; updated ${Array.isArray(rows) ? rows.length : 0}.`,
      );
    }
  }
}
