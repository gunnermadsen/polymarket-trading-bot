import { MigrationInterface, QueryRunner } from 'typeorm';

export class CorrectLiveDynamicFees1786929929000
  implements MigrationInterface
{
  name = 'CorrectLiveDynamicFees1786929929000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`SET LOCAL lock_timeout = '3s';`);
    await queryRunner.query(`SET LOCAL statement_timeout = '60s';`);
    await queryRunner.query(`
      WITH corrected AS MATERIALIZED (
        SELECT
          fill.fill_id,
          fill.timestamp_utc,
          round(
            fill.size
              * (orders.raw_payload #>> '{request,metadata,dynamic_fee_rate}')::numeric
              * fill.price
              * (1 - fill.price),
            10
          )::numeric(30,10) AS corrected_fee
        FROM polymarket.fills fill
        JOIN polymarket.orders orders
          ON orders.order_id = fill.order_id
         AND orders.process_id = fill.process_id
        WHERE fill.source = 'live'
          AND orders.raw_payload #>> '{request,metadata,dynamic_fee_rate}'
              ~ '^[0-9]+([.][0-9]+)?$'
          AND (orders.raw_payload #>> '{request,metadata,dynamic_fee_rate}')::numeric > 0
          AND (orders.raw_payload #>> '{request,metadata,dynamic_fee_rate}')::numeric <= 1
          AND fill.size > 0
          AND fill.price > 0
          AND fill.price < 1
      )
      UPDATE polymarket.fills fill
      SET fee = corrected.corrected_fee
      FROM corrected
      WHERE fill.fill_id = corrected.fill_id
        AND fill.timestamp_utc = corrected.timestamp_utc
        AND fill.fee IS DISTINCT FROM corrected.corrected_fee;
    `);
    await queryRunner.query(`
      WITH settlement_fees AS MATERIALIZED (
        SELECT
          settlement.settlement_id,
          round(SUM(fill.fee), 10)::numeric(30,10) AS entry_fees
        FROM polymarket.btc_paper_settlement_ledger settlement
        JOIN LATERAL jsonb_array_elements_text(settlement.fill_ids) fill_id(value)
          ON true
        JOIN polymarket.fill_identities identity
          ON identity.fill_id = fill_id.value::uuid
         AND identity.process_id = settlement.process_id
        JOIN polymarket.fills fill
          ON fill.fill_id = identity.fill_id
         AND fill.timestamp_utc = identity.timestamp_utc
         AND fill.process_id = settlement.process_id
         AND fill.order_id = settlement.order_id
         AND fill.source = 'live'
        WHERE settlement.execution_mode = 'live'
        GROUP BY settlement.settlement_id
      )
      UPDATE polymarket.btc_paper_settlement_ledger settlement
      SET entry_fees = settlement_fees.entry_fees,
          net_pnl = (
            settlement.payout - settlement.entry_notional - settlement_fees.entry_fees
          )::numeric(30,10),
          updated_at = now()
      FROM settlement_fees
      WHERE settlement.settlement_id = settlement_fees.settlement_id
        AND (
          settlement.entry_fees <> settlement_fees.entry_fees
          OR settlement.net_pnl <>
            settlement.payout - settlement.entry_notional - settlement_fees.entry_fees
        );
    `);
  }

  public async down(): Promise<void> {
    // Financial evidence corrections are intentionally irreversible: the previous zero fees were
    // not venue truth and restoring them would knowingly corrupt realized PnL.
  }
}
