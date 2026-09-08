"""Regenerate deterministic frozen feature parity cases using training's own formulas."""
import json, math
from datetime import datetime,timedelta,timezone
from pathlib import Path
import polars as pl
from btc_directional_model.core_features import derive_core_point_in_time_features, derive_oracle_point_in_time_features, attach_causal_oracle_rounds

root=Path(__file__).parents[1]
model=json.loads(next((root/'runtime-models').glob('btc-5m-extended-specialist-official-umr-*/model.json')).read_text())
names=model['features']['names'][:67]+model['features']['names'][75:79]
start=datetime(2026,8,20,12,0,tzinfo=timezone.utc)
raw=[];klines=[]
for second in range(300):
 at=start+timedelta(seconds=second)
 price=100000+20*math.sin(second/7)+second*.17
 op=100000+20*math.sin((second-1)/7)+(second-1)*.17
 q=1000+second*7; buy=q*(.4+.15*math.sin(second/11))
 raw.append(dict(market_id='umr-fixture',window_start=start,observed_at=at,seconds_elapsed=second,opening_boundary=100000+20*math.sin(-1/7)-.17,
    btc_open=op,btc_high=max(price,op)+2,btc_low=min(price,op)-2,btc_close=price,btc_quote_volume=q,btc_taker_buy_quote_volume=buy,trade_count=10+second%9))
 klines.append(dict(open_timestamp=(at-timedelta(seconds=1)).isoformat(),close_timestamp=at.isoformat(),open_price=str(op),high_price=str(max(price,op)+2),low_price=str(min(price,op)-2),close_price=str(price),base_volume=str(q/price),quote_volume=str(q),trade_count=10+second%9,taker_buy_base_volume=str(buy/price),taker_buy_quote_volume=str(buy),first_aggregate_trade_id=second*100,last_aggregate_trade_id=second*100+20,first_source_timestamp=(at-timedelta(seconds=1)).isoformat(),last_source_timestamp=(at-timedelta(microseconds=1)).isoformat(),max_received_at=at.isoformat(),source_complete=True,synthetic=False))
oracle=[];runtime_oracle=[]
for index,second in enumerate(range(-120,301,30)):
 at=start+timedelta(seconds=second);price=100003+second*.1
 oracle.append(dict(oracle_phase_id=1,oracle_round_id=index+1,oracle_source_timestamp=at-timedelta(seconds=1),oracle_block_timestamp=at,oracle_block_number=1000+index,oracle_log_index=0,oracle_price=price))
 runtime_oracle.append(dict(phase_id=1,aggregator_round_id=index+1,source_timestamp=(at-timedelta(seconds=1)).isoformat(),block_timestamp=at.isoformat(),block_number=1000+index,log_index=0,price=str(price),available_at=at.isoformat()))
f=derive_oracle_point_in_time_features(derive_core_point_in_time_features(attach_causal_oracle_rounds(pl.DataFrame(raw),pl.DataFrame(oracle)))).filter(pl.col('seconds_elapsed').is_between(60,85)&(pl.col('seconds_elapsed')%5==0))
rows=[]
for row in f.iter_rows(named=True):
 seconds=row['seconds_elapsed']
 values=[row[n] for n in names]
 rows.append(dict(seconds=seconds,window_start=start.isoformat(),names=json.dumps(names),klines=json.dumps(klines[:seconds+1]),oracle=json.dumps(runtime_oracle),expected=json.dumps(values,allow_nan=False)))
pl.DataFrame(rows).write_parquet(root/'tests/fixtures/umr-core-feature-parity.parquet')
print('Wrote',len(rows),'raw-to-feature parity cases')
