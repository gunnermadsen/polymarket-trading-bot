# Validate the provisioned Binance L2 persistence expression using Prometheus.
require 'yaml'
require 'open3'

root = File.expand_path('..', __dir__)
path = "#{root}/common/configs/grafana/provisioning/alerting/rules-market-data-ingester.yml"
rules_document = if YAML.respond_to?(:unsafe_load_file)
                   YAML.unsafe_load_file(path)
                 else
                   YAML.load_file(path)
                 end
rule = rules_document.fetch('groups').flat_map { |group| group.fetch('rules') }
  .find { |candidate| candidate['uid'] == 'mdi_binance_l2_persistence_stale' }
raise 'missing Binance L2 persistence alert' unless rule
raise 'unexpected no-data behavior' unless rule['noDataState'] == 'OK'
raise 'critical severity was not preserved' unless rule.dig('labels', 'severity') == 'critical'

expression = rule.fetch('data').first.dig('model', 'expr')
raise 'staleness comparison must preserve healthy zeroes' unless expression.include?('> bool 60')

persistence = 'market_data_ingester_strategy_last_persistence_timestamp_seconds{strategy="binance_spot_btcusdt_l2_snapshots"}'
desired = 'market_data_ingester_strategy_state{strategy="binance_spot_btcusdt_l2_snapshots",desired_state="running",observed_state="running",health_status="healthy"}'
series = ->(name, values) { {'series' => name, 'values' => values} }
test = ->(name, inputs, expected) do
  {
    'name' => name,
    'interval' => '10s',
    'input_series' => inputs,
    'promql_expr_test' => [{
      'expr' => "sum(#{expression}) or vector(0)",
      'eval_time' => '2m',
      'exp_samples' => [{'labels' => '{}', 'value' => expected}]
    }]
  }
end

fixture = {
  'evaluation_interval' => '10s',
  'tests' => [
    test.call('fresh persistence is normal', [series.call(persistence, '110+0x20'), series.call(desired, '1+0x20')], 0),
    test.call('stale desired-running persistence alerts', [series.call(persistence, '30+0x20'), series.call(desired, '1+0x20')], 1),
    test.call('strategy not desired-running is normal', [series.call(persistence, '30+0x20')], 0)
  ]
}

promtool_image = ENV['PROMTOOL_DOCKER_IMAGE']
command = if promtool_image && !promtool_image.empty?
            ['docker', 'run', '--rm', '-i', '--entrypoint', 'promtool', promtool_image,
             'test', 'rules', '/dev/stdin']
          else
            ['docker', 'exec', '-i', 'prometheus', 'promtool', 'test', 'rules', '/dev/stdin']
          end
output, status = Open3.capture2e(*command, stdin_data: YAML.dump(fixture))
puts output
abort 'Prometheus expression tests failed' unless status.success?
puts 'PASS Binance L2 persistence alert fixtures'
