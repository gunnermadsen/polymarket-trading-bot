# Validate the provisioned expressions using Prometheus itself; no running configuration changes.
require 'yaml'
require 'json'
require 'open3'
require 'fileutils'
root = File.expand_path('..', __dir__)
rules = Dir["#{root}/common/configs/grafana/provisioning/alerting/*.yml"].flat_map { |f| YAML.load_file(f).fetch('groups', []).flat_map { |g| g['rules'] } }
raise 'duplicate alert UID' unless rules.map { |r| r['uid'] }.uniq.size == rules.size
selected = %w[execution_freshness_violation execution_freshness_monitor_missing clob_book_source_stale_for_execution clob_book_unavailable_sustained binance_reference_unavailable].to_h { |uid| [uid, rules.find { |r| r['uid'] == uid } || (raise "missing #{uid}")] }
selected.each_value do |r|
  raise "unsafe no-data behavior: #{r['uid']}" unless r['noDataState'] == 'Alerting' && r['execErrState'] == 'Error' && !r['isPaused']
  raise 'obsolete runtime input' unless r['data'][0]['datasourceUid'] == 'prometheus'
end
entry = File.read("#{root}/common/scripts/grafana-provisioning-entrypoint.sh")
raise 'rules not copied by provisioning' unless entry.include?('cp "${SRC_DIR}/alerting/rules-execution-freshness.yml"')
prefix = 'polymarket_execution_freshness_'
labels = '{process_id="p1",execution_mode="paper",job="polymarket-bot",instance="bot"}'
series = ->(name, values) { {'series'=>name, 'values'=>values} }
base = [series.call(prefix+'exporter_ready{job="polymarket-bot",instance="bot"}', '1+0x20'), series.call('up{job="polymarket-bot",instance="bot"}', '1+0x20')]
monitor = %w[expected monitor_ready executions_total violations_total book_rejections_total reference_rejections_total].map { |n| series.call(prefix+n+labels, %w[expected monitor_ready].include?(n) ? '1+0x20' : '0+0x20') }
expression = ->(uid) { selected.fetch(uid)['data'][0]['model']['expr'] }
tests = []
add = ->(name, input, uid, expected, at='5m') do
  tests << {'name'=>name, 'interval'=>'1m', 'input_series'=>input, 'promql_expr_test'=>[{'expr'=>"sum(#{expression.call(uid)}) or vector(0)", 'eval_time'=>at, 'exp_samples'=>[{'labels'=>'{}','value'=>expected}]}]}
end
add.call('idle process initialized, no executions is healthy',base+monitor,'execution_freshness_monitor_missing',0)
add.call('no active process is healthy',base,'execution_freshness_monitor_missing',0)
add.call('missing process instrumentation is visible',base+[monitor[0]],'execution_freshness_monitor_missing',1)
add.call('missing counter cannot be hidden by monitor-ready',base+monitor.reject { |s| s['series'].start_with?(prefix+'violations_total') },'execution_freshness_monitor_missing',1)
add.call('exporter missing with scrape still up',base.drop(1),'execution_freshness_monitor_missing',1)
add.call('zero startup counter is not a violation',base+[series.call(prefix+'last_violation_timestamp_seconds'+labels,'0+0x20')],'execution_freshness_violation',0)
add.call('first observed violation fires without needing a prior scrape',base+[series.call(prefix+'last_violation_timestamp_seconds'+labels,'60+0x20')],'execution_freshness_violation',1)
add.call('incident window expires automatically',base+[series.call(prefix+'last_violation_timestamp_seconds'+labels,'60+0x20')],'execution_freshness_violation',0,'12m')
add.call('fresh executions do not cause rejection alerts',base+monitor,'clob_book_source_stale_for_execution',0)
# Gauge recovery verifies a restarted monitor is healthy without any new orders.
recovered = monitor.map { |s| s.merge('values'=>s['series'].start_with?(prefix+'monitor_ready') ? '_ _ 1+0x18' : s['values']) }
add.call('monitor recovers without manual reenable or trades',base+recovered,'execution_freshness_monitor_missing',0)
add.call('unready canonical book product',base+monitor+[series.call('polymarket_market_data_product_ready{product="polymarket_btc_five_minute_orderbooks"}','0+0x20')],'clob_book_unavailable_sustained',1)
add.call('ready canonical book product',base+monitor+[series.call('polymarket_market_data_product_ready{product="polymarket_btc_five_minute_orderbooks"}','1+0x20')],'clob_book_unavailable_sustained',0)
add.call('missing selected book product is not healthy',base+monitor,'clob_book_unavailable_sustained',1)
add.call('no process means no required book product',base,'clob_book_unavailable_sustained',0)
add.call('ready Binance reference product',base+monitor+[series.call('polymarket_market_data_product_ready{product="binance_spot_btcusdt_one_second_ohlcv"}','1+0x20')],'binance_reference_unavailable',0)
add.call('missing selected Binance reference is not healthy',base+monitor,'binance_reference_unavailable',1)
# Test the actual threshold expression instead of an assumed count/increase extrapolation.
reject_rule = selected.fetch('clob_book_source_stale_for_execution')
threshold = reject_rule['data'].last['model']['conditions'][0]['evaluator']['params'][0]
raise 'unexpected rejection alert persistence' unless reject_rule['for'] == '60s'
reject_series = [series.call(prefix+'book_rejections_total'+labels,'0+3x20'),series.call(prefix+'reference_rejections_total'+labels,'0+0x20')]
tests << {'name'=>'repeated prevented executions cross the configured warning threshold','interval'=>'1m','input_series'=>base+reject_series,'promql_expr_test'=>[{'expr'=>"sum((#{expression.call('clob_book_source_stale_for_execution')}) > bool #{threshold})",'eval_time'=>'5m','exp_samples'=>[{'labels'=>'{}','value'=>1}]}]}
fixture = {'evaluation_interval'=>'1m','tests'=>tests}
dir = "#{root}/target/execution-freshness-alerts"
FileUtils.mkdir_p(dir)
File.write("#{dir}/prometheus-tests.yml", YAML.dump(fixture))
output, status = Open3.capture2e('docker','exec','-i','prometheus','promtool','test','rules','/dev/stdin',stdin_data:YAML.dump(fixture))
puts output
abort 'Prometheus expression tests failed' unless status.success?
puts "PASS #{tests.length} provisioned alert expression fixtures and missing-data/provisioning checks"
