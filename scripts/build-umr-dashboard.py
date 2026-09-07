"""Build the provisioned UMR dashboard from stable monitoring contracts."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DS={'type':'prometheus','uid':'prometheus'}
PANELS=[];Y=0
S='process_id=~"$process"'
def metric(name):return 'polymarket_umr_'+name+'{'+S+'}'
def count(name,reason=None):
 labels=S+(',reason="'+reason+'"' if reason else '')
 return 'sum by (process_id) (polymarket_umr_'+name+'_total{'+labels+'})'
def rate(name):return 'sum by (process_id) (rate(polymarket_umr_'+name+'_total{'+S+'}[$__rate_interval]))'
def labelled(expr):return '('+expr+') * on(process_id) group_left(model_key) polymarket_umr_model_info{'+S+'}'
def row(title):
 global Y
 PANELS.append(dict(id=len(PANELS)+1,type='row',title=title,collapsed=False,panels=[],gridPos=dict(x=0,y=Y,w=24,h=1)));Y+=1

def panel(title,expr,description,unit='short',kind='timeseries',x=0,w=12,h=8,legend='{{model_key}}',colour='palette-classic',aggregate=False):
 d=dict(id=len(PANELS)+1,title=title,type=kind,datasource=DS,description=description,gridPos=dict(x=x,y=Y,w=w,h=h),
  targets=[dict(refId='A',expr=expr if aggregate else labelled(expr),legendFormat=title if aggregate else legend,range=kind=='timeseries',instant=kind!='timeseries')],
  fieldConfig=dict(defaults=dict(unit=unit,decimals=2,color=dict(mode=colour),noValue='Awaiting evidence',thresholds=dict(mode='absolute',steps=[dict(color='blue',value=None)])),overrides=[]))
 if kind=='timeseries':
  d['fieldConfig']['defaults']['custom']=dict(drawStyle='line',lineWidth=2,fillOpacity=8,showPoints='never',spanNulls=False,axisLabel='',axisPlacement='auto',scaleDistribution=dict(type='linear'))
  d['options']=dict(legend=dict(displayMode='table',placement='bottom',calcs=['lastNotNull']),tooltip=dict(mode='multi',sort='desc'))
 elif kind=='stat': d['options']=dict(reduceOptions=dict(calcs=['lastNotNull'],fields='',values=False),orientation='auto',textMode='value_and_name',colorMode='none',graphMode='none',justifyMode='auto',wideLayout=True)
 elif kind=='table':d['options']=dict(showHeader=True,cellHeight='md',footer=dict(show=False))
 PANELS.append(d);return d

PANELS.append(dict(id=1,type='text',title='Unified Model Runtime · Live model operations',gridPos=dict(x=0,y=0,w=24,h=5),options=dict(mode='markdown',content='Follow each model from **data → prediction → admission → fill → settlement**. Select one process to investigate or compare the full collection.\n\n**Scope:** operational rates follow the selected time range. Economic and calibration aggregates cover the current instrumentation session and reset on service/model/run/config changes. They are not lifetime accounting. Missing outcomes remain **Awaiting evidence**, never artificial zeros.')));Y=5
row('At a glance')
for i,(title,expr,unit,desc) in enumerate([
 ('Ready processes','sum('+metric('runtime_ready')+')','short','Latest process readiness; feed and feature gaps recover automatically.'),
 ('Oldest inference','time() - min('+metric('last_success_timestamp_seconds')+')','s','Elapsed time since successful inference. Entry schedules naturally create quiet periods.'),
 ('Inference rate','sum('+rate('inferences')+')','ops','Successful and failed inference attempts per second; inspect the error panel below.'),
 ('Session net PnL','sum('+metric('realized_pnl_usd')+')','currencyUSD','Net PnL recognized by authoritative settlement during this instrumentation session.'),
 ]):panel(title,expr,desc,unit,'stat',i*6,6,5,aggregate=True)
Y+=5
row('Runtime health and immutable identity')
p=panel('Selected models',metric('enabled'),'Model identity is pinned by artifact and feature checksums. Discovery never changes an active selection.','short','table',0,24,8)
p['targets']=[dict(refId='A',expr='polymarket_umr_model_info{'+S+'}',format='table',instant=True)]
p['transformations']=[dict(id='organize',options=dict(excludeByName={'Time':True,'Value':True,'__name__':True,'job':True,'instance':True},renameByName={'process_id':'Process','model_key':'Model','execution_mode':'Mode','artifact_sha256':'Artifact SHA-256','feature_schema_sha256':'Feature SHA-256','config_hash':'Configuration SHA-256'}))]
Y+=8
panel('Readiness over time',metric('runtime_ready'),'A transient unavailable input must not permanently disable the process.','bool',x=0)
panel('Observation heartbeat age','time() - '+metric('last_observation_timestamp_seconds'),'Low age confirms callbacks are reaching the process even outside model entry windows.','s',x=12);Y+=8
row('Opportunity and decision funnel')
for i,(title,expr,desc) in enumerate([
 ('Scheduled opportunities',count('opportunities'),'Claimed opportunities at the configured candidate cadence. Pre-claim skips are displayed separately.'),
 ('Inferred markets',count('markets','inferred'),'Distinct markets with at least one prediction in the instrumentation session.'),
 ('Admitted markets',count('markets','admitted'),'Distinct markets admitted by the model; execution can still reject an order.'),
 ('Filled orders',count('execution','filled'),'Orders acknowledged as filled by the existing execution pathway.'),
 ]):
 panel(title,expr,desc,kind='stat',x=(i%2)*12,w=12,h=8)
 if i%2:Y+=8
p=panel('Skipped opportunities by reason','sum by(process_id,reason) (rate(polymarket_umr_skipped_total{'+S+'}[$__rate_interval]))','Includes absent history, persistence unavailability, duplicate candidate claims and out-of-schedule callbacks. Repeated callbacks are not distinct market opportunities.',x=0,legend='{{model_key}} · {{reason}}')
panel('Persisted decision stages','sum by(process_id,reason) (rate(polymarket_umr_decisions_total{'+S+'}[$__rate_interval]))','Stages are lifecycle transitions. Do not sum them as mutually exclusive outcomes.',x=12,legend='{{model_key}} · {{reason}}');Y+=8
panel('Market inference coverage',count('markets','inferred')+' / '+count('markets','observed'),'Distinct inferred markets / observed markets. Current-session denominator includes observed markets without inference.','percentunit',x=0)
panel('Market admission coverage',count('markets','admitted')+' / '+count('markets','observed'),'Distinct admitted markets / observed markets. Coverage across models overlaps and must not be summed.','percentunit',x=12);Y+=8
panel('Eligible market coverage',count('markets','inferred')+' / '+count('markets','eligible'),'Inferred markets / markets with at least one claimed candidate slot. Feed outages before a candidate can be claimed are visible in observed-market coverage and skips.','percentunit',x=0)
panel('Execution transitions','sum by(process_id,reason)(rate(polymarket_umr_execution_total{'+S+'}[$__rate_interval]))','Order states from the existing execution engine, separate from persisted decision stages.','ops',x=12,legend='{{model_key}} · {{reason}}');Y+=8
row('Inputs and latency')
panel('Missing feature fraction',metric('missing_feature_fraction'),'Fraction of model inputs using native missing values. Optional absence remains visible; similar source names do not establish equivalence.','percentunit',x=0)
panel('Feature age',metric('feature_age_seconds'),'Age of the immutable feature snapshot at the process observation time.','s',x=12);Y+=8
panel('Stage latency · p95','histogram_quantile(0.95, sum by(process_id,stage,le)(rate(polymarket_umr_stage_duration_seconds_bucket{'+S+'}[$__rate_interval])))','Stages include features, inference, strategy and complete observation. Durations overlap; do not sum percentiles.','s',x=0,legend='{{model_key}} · {{stage}}')
panel('Feature and inference errors','sum by(process_id) (rate(polymarket_umr_feature_builds_total{'+S+',reason="error"}[$__rate_interval])) + sum by(process_id) (rate(polymarket_umr_inferences_total{'+S+',reason="error"}[$__rate_interval]))','Failures remain distinct from a model correctly abstaining.','ops',x=12);Y+=8
panel('Runtime block reasons','sum by(process_id,reason)(rate(polymarket_umr_readiness_blocks_total{'+S+'}[$__rate_interval]))','Shared infrastructure readiness is filtered for each process’s actual requirements.','ops',x=0,legend='{{model_key}} · {{reason}}')
panel('Telemetry capacity drops','sum by(process_id)(increase(polymarket_umr_telemetry_dropped_total{'+S+'}[$__range]))','Dropped unresolved prediction records limit analytical coverage; operational trading continues.','short',x=12);Y+=8
panel('Feature failure reasons','sum by(process_id,reason)(rate(polymarket_umr_feature_failures_total{'+S+'}[$__rate_interval]))','Controlled error categories; diagnostic detail is in the process logs.','ops',x=0,legend='{{model_key}} · {{reason}}')
panel('Inference failure reasons','sum by(process_id,reason)(rate(polymarket_umr_inference_failures_total{'+S+'}[$__rate_interval]))','A failed score is distinct from an intentional model abstention.','ops',x=12,legend='{{model_key}} · {{reason}}');Y+=8
for i,q in enumerate([0.5,0.99]):
 panel('Inference latency · p'+str(int(q*100)),'histogram_quantile('+str(q)+', sum by(process_id,le)(rate(polymarket_umr_stage_duration_seconds_bucket{'+S+',stage="inference"}[$__rate_interval])))','Inference latency percentile. Native host smoke results do not establish deployed container capacity.','s',x=i*12)
Y+=8
panel('Inference failure ratio',count('inferences','error')+' / '+count('inferences'),'Failed inference attempts / all attempts in this instrumentation session.','percentunit',x=0)
panel('Feature failure ratio',count('feature_builds','error')+' / '+count('feature_builds'),'Failed feature attempts / all feature attempts in this instrumentation session.','percentunit',x=12);Y+=8
row('Prediction and admission behavior')
panel('Latest probability · UP',metric('probability_up'),'Probability of UP, independent of whether admission or execution allowed a trade.','percentunit',x=0)
panel('Latest confidence',metric('confidence'),'Maximum of UP and DOWN probability. High confidence alone does not establish good entry economics.','percentunit',x=12);Y+=8
panel('Model admission decisions','sum by(process_id,reason)(rate(polymarket_umr_model_admission_total{'+S+'}[$__rate_interval]))','Accepted/rejected model decisions; independent of strategy, capital and execution gates.','ops',x=0,legend='{{model_key}} · {{reason}}')
panel('Admission rejection reasons','sum by(process_id,reason)(rate(polymarket_umr_admission_reasons_total{'+S+'}[$__rate_interval]))','Frozen adapter reasons distinguish cost, confidence, expected edge, learned loss and temporal disagreement.','ops',x=12,legend='{{model_key}} · {{reason}}');Y+=8
panel('Learned admission probability',metric('admission_probability'),'Only defined for models with learned admission. Empty for programmatic models.','percentunit',x=0)
panel('Predicted loss severity',metric('predicted_loss'),'Frozen learned output, clipped at zero exactly as training. This is not realized loss.','currencyUSD',x=12);Y+=8
panel('Temporal probability dispersion',metric('temporal_std'),'Population standard deviation across the frozen predictor and temporal snapshots. Only defined for consensus models.','short',x=0)
panel('Temporal direction agreement',metric('temporal_agreement'),'Fraction of frozen components agreeing with the official predictor.','percentunit',x=12);Y+=8
panel('UP / DOWN / ABSTAIN','sum by(process_id,reason)(increase(polymarket_umr_actions_total{'+S+'}[$__range]))','Model actions over the selected range. Raw directional probabilities remain available for abstained candidates.','short',x=0,legend='{{model_key}} · {{reason}}')
panel('Predicted stress edge',metric('predicted_stress_edge'),'Frozen learned stress-edge output per share; an estimate, not realized stress PnL.','currencyUSD',x=12);Y+=8
for i,(name,title) in enumerate([('probability_bin','UP probability distribution'),('confidence_bin','Confidence distribution')]):
 p=panel(title,'sum by(process_id,reason)(increase(polymarket_umr_'+name+'_total{'+S+'}[$__range]))','Counts per decile: bin 0 is [0,0.1), bin 9 is [0.9,1]. These are inference opportunities, not independent markets.','short','table',x=i*12,legend='{{model_key}} · decile {{reason}}')
 p['targets'][0]['format']='table'
Y+=8
panel('Strategy rejection reasons','sum by(process_id,reason)(rate(polymarket_umr_strategy_rejections_total{'+S+'}[$__rate_interval]))','Existing strategy guards may reject an admitted model prediction.','ops',x=0,legend='{{model_key}} · {{reason}}')
panel('Failed observation callbacks','sum by(process_id)(rate(polymarket_umr_observations_failed_total{'+S+'}[$__rate_interval]))','Unhandled observation errors, including persistence and execution failures, retain existing recovery behavior.','ops',x=12);Y+=8
row('Resolved prediction quality · evaluation weighted')
panel('Brier score',metric('brier_sum')+' / '+metric('brier_count'),'Mean squared probability error over resolved inference opportunities, including rejected trades. Multiple predictions per market are correlated.','short',x=0)
panel('Directional correctness',count('prediction_outcomes','correct')+' / '+count('prediction_outcomes'),'Correct resolved predictions / all resolved predictions. Not the executed-trade win rate.','percentunit',x=12);Y+=8
p=panel('Calibration by probability bin','polymarket_umr_calibration_up_outcomes{'+S+'} / polymarket_umr_calibration_count{'+S+'}','Observed UP frequency for each decile. Compare with mean predicted probability; bins with no outcomes remain undefined.','percentunit','table',0,12,8,legend='{{model_key}} · bin {{bin}}')
p['targets'][0]['format']='table'
panel('Unresolved prediction records',metric('pending_predictions'),'Pending official outcomes retained in bounded memory. Restarts reset this operational buffer; durable feature/decision records remain.','short',x=12);Y+=8
p=panel('Calibration mean prediction','polymarket_umr_calibration_probability_sum{'+S+'} / polymarket_umr_calibration_count{'+S+'}','Mean forecast probability in each resolved decile; compare with observed UP frequency.','percentunit','table',x=0)
p['targets'][0]['format']='table'
p=panel('Calibration sample counts','polymarket_umr_calibration_count{'+S+'}','Number of resolved inference opportunities in each decile. A sparse bin cannot support a calibration conclusion.','short','table',x=12)
p['targets'][0]['format']='table';Y+=8
panel('Resolved prediction sample count',metric('brier_count'),'Evaluation-weighted denominator for Brier and correctness.','short',x=0)
panel('Settled trade win rate',count('trade_outcomes','win')+' / ('+count('trade_outcomes','win')+' + '+count('trade_outcomes','loss')+')','Winning settlements / win-or-loss settlements; sample counts are displayed below.','percentunit',x=12);Y+=8
row('Execution and economics · current session')
for i,(title,expr,unit,desc) in enumerate([
 ('Expectancy / trade',metric('realized_pnl_usd')+' / '+count('trade_outcomes'),'currencyUSD','Recognized net PnL / settled trades.'),
 ('Profit factor',metric('gross_profit_usd')+' / '+metric('gross_loss_usd'),'short','Gross positive settled PnL / absolute gross negative settled PnL.'),
 ('Wins to recover one loss','('+metric('gross_loss_usd')+' / '+count('trade_outcomes','loss')+') / ('+metric('gross_profit_usd')+' / '+count('trade_outcomes','win')+')','short','Average absolute losing trade / average winning trade. Undefined until both exist.'),
 ('Maximum drawdown',metric('max_drawdown_usd'),'currencyUSD','Peak-to-trough recognized net PnL within the current instrumentation session.'),
 ]):
 panel(title,expr,desc,unit,'stat',(i%2)*12,12,8)
 if i%2:Y+=8
panel('Realized net PnL',metric('realized_pnl_usd'),'Net PnL from successfully recognized settlements; excludes unresolved positions.','currencyUSD',x=0)
panel('Settled trade outcomes','sum by(process_id,reason)(polymarket_umr_trade_outcomes_total{'+S+'})','Economic win/loss/push classification by realized net PnL.','short',x=12,legend='{{model_key}} · {{reason}}');Y+=8
panel('Average entry share price',metric('fill_notional_usd')+' / '+metric('filled_shares'),'Actual fill notional / actual shares filled. Fees are displayed separately.','currencyUSD',x=0)
panel('Average entry second',metric('entry_seconds_sum')+' / '+metric('entry_fill_count'),'Average market age of filled entry decisions. Fill-event weighted.','s',x=12);Y+=8
panel('Quote-to-fill slippage / share',metric('slippage_notional_usd')+' / '+metric('filled_shares'),'Actual fill minus executable quote at decision time, weighted by shares. Positive is adverse.','currencyUSD',x=0)
panel('Actual fill fees',metric('fill_fees_usd'),'Fees reported by fills in the current session.','currencyUSD',x=12);Y+=8
panel('Settled fees',metric('fees_usd'),'Entry fees associated with settlements recognized during this session; differs from fees on still-open fills.','currencyUSD',x=0)
panel('Registered telemetry capacity drops','scalar(polymarket_umr_registry_dropped_total) + 0 * '+metric('enabled'),'Global registration capacity losses, repeated per selected process for visibility. Capacity losses never authorize or block orders.','short',x=12);Y+=8
row('Investigation')
PANELS.append(dict(id=len(PANELS)+1,title='Process lifecycle and diagnostics',type='logs',datasource={'type':'loki','uid':'loki'},gridPos=dict(x=0,y=Y,w=24,h=12),description='Process IDs remain structured fields rather than Loki stream labels. This panel also includes existing process-scoped runtime diagnostics.',targets=[dict(refId='A',expr='{service_name="polymarket-bot"} | json | fields_process_id =~ "$process"')],options=dict(showTime=True,showLabels=False,showCommonLabels=False,wrapLogMessage=True,prettifyLogMessage=True,enableLogDetails=True,sortOrder='Descending',dedupStrategy='none')))
D=dict(uid='unified-model-runtime',title='Unified Model Runtime',tags=['UMR','models','trading'],timezone='browser',schemaVersion=39,version=1,editable=False,refresh='10s',time=dict(**{'from':'now-6h','to':'now'}),graphTooltip=1,
 annotations=dict(list=[dict(builtIn=1,datasource=dict(type='grafana',uid='-- Grafana --'),enable=True,hide=True,iconColor='rgba(0, 211, 255, 1)',name='Annotations & Alerts',type='dashboard')]),
 templating=dict(list=[dict(name='process',label='Trading process',type='query',datasource=DS,query=dict(query='label_values(polymarket_umr_model_info, process_id)',refId='StandardVariableQuery'),definition='label_values(polymarket_umr_model_info, process_id)',refresh=1,multi=True,includeAll=True,allValue='.*',current=dict(text='All',value='$__all'),options=[],sort=1)]),panels=PANELS)
path=ROOT/'common/configs/grafana/dashboards/unified-model-runtime.json'
path.write_text(json.dumps(D,indent=2)+'\n')
print(len(PANELS),'panels, no PostgreSQL datasource')
