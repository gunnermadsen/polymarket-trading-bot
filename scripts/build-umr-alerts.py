"""Provision sample-aware UMR alerts using the existing Grafana/Prometheus stack."""
import json
from pathlib import Path
DS={'type':'prometheus','uid':'prometheus'}
def rule(uid,title,expr,wait='2m',severity='warning',description=''):
 return dict(uid=uid,title=title,condition='C',data=[
  dict(refId='A',datasourceUid='prometheus',relativeTimeRange={'from':600,'to':0},model=dict(datasource=DS,editorMode='code',expr=expr,instant=True,range=False,refId='A',intervalMs=1000,maxDataPoints=43200)),
  dict(refId='B',datasourceUid='__expr__',relativeTimeRange={'from':0,'to':0},model=dict(datasource={'type':'__expr__','uid':'__expr__'},expression='A',reducer='last',settings={'mode':'dropNN'},type='reduce',refId='B')),
  dict(refId='C',datasourceUid='__expr__',relativeTimeRange={'from':0,'to':0},model=dict(datasource={'type':'__expr__','uid':'__expr__'},expression='B',type='threshold',refId='C',conditions=[dict(evaluator={'params':[0],'type':'gt'},operator={'type':'and'},query={'params':['C']},reducer={'params':[],'type':'last'},type='query')]))],
  noDataState='OK',execErrState='Error',**{'for':wait},annotations=dict(summary=title,description=description,dashboardUid='unified-model-runtime'),labels=dict(family='umr',service='polymarket-bot',severity=severity,environment='$GRAFANA_ALERT_ENVIRONMENT'),isPaused=False)
rules=[
 rule('umr_bot_scrape_missing','UMR bot telemetry unavailable','absent(up{job="polymarket-bot"}) or (up{job="polymarket-bot"} == bool 0)',description='The bot scrape is missing or failing. Check service and monitoring availability; do not infer that process inactivity is healthy.'),
 rule('umr_runtime_unready','UMR enabled process input readiness unavailable','(polymarket_umr_runtime_ready == bool 0) and on(process_id) (polymarket_umr_enabled == 1)',wait='5m',description='The selected model has remained unavailable across a full market window. Inspect required inputs and native missing coverage. Recovery remains automatic.'),
 rule('umr_callback_stalled','UMR enabled process callbacks stalled','(time() - polymarket_umr_last_observation_timestamp_seconds > bool 60) and on(process_id) (polymarket_umr_enabled == 1)',description='An enabled runtime registration has no observation callback for over one minute. This alert does not disable trading.'),
 rule('umr_inference_stalled','UMR scheduled opportunities without inference','(sum by(process_id)(increase(polymarket_umr_opportunities_total[10m])) > bool 0) * on(process_id) (sum by(process_id)(increase(polymarket_umr_inferences_total{reason="success"}[10m])) == bool 0)',description='Opportunities were claimed during the last ten minutes, but no prediction completed. Inspect feature and history availability.'),
 rule('umr_feature_errors','UMR feature construction repeatedly unavailable','sum by(process_id)(increase(polymarket_umr_feature_builds_total{reason="error"}[10m])) > bool 5',description='At least six recent feature attempts failed. Inspect missing inputs, book causality and history; optional native missing values are not feature errors.'),
 rule('umr_inference_errors','UMR repeated inference failures','sum by(process_id)(increase(polymarket_umr_inferences_total{reason="error"}[10m])) > bool 2',description='Inference returned errors on multiple opportunities. An intentional admission rejection does not count as failure.'),
 rule('umr_telemetry_dropped','UMR analytical evidence dropped','sum by(process_id)(increase(polymarket_umr_telemetry_dropped_total[10m])) > bool 0',description='Bounded telemetry capacity was exceeded. Quality metrics have incomplete coverage; durable records and trading controls remain authoritative.'),
 rule('umr_quality_review','UMR prediction quality needs review','((polymarket_umr_brier_sum / polymarket_umr_brier_count) > bool 0.25) and on(process_id) (polymarket_umr_brier_count >= 100)',wait='10m',description='Session Brier exceeds 0.25 after at least 100 resolved evaluations. Evaluations within a market are correlated; investigate source coverage and calibration before interpreting significance.'),
 rule('umr_expectancy_review','UMR realized expectancy needs review','(polymarket_umr_realized_pnl_usd < bool 0) and on(process_id) (sum by(process_id)(polymarket_umr_trade_outcomes_total) >= 50)',wait='10m',description='Session realized net PnL is negative after at least fifty settled trades. This is a review notification, not an automated trading kill switch.'),
]
p=Path(__file__).resolve().parents[1]/'common/configs/grafana/provisioning/alerting/rules-umr.yml'
p.write_text(json.dumps(dict(apiVersion=1,groups=[dict(orgId=1,name='unified-model-runtime',folder='Unified Model Runtime',interval='30s',rules=rules)]),indent=2)+'\n')
print(len(rules),'provisioned alerts')
