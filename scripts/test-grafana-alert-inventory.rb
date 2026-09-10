# Validate that the checked-in Grafana alert inventory is complete and self-consistent.
require 'yaml'
require 'open3'

root = File.expand_path('..', __dir__)
alert_dir = File.join(root, 'common/configs/grafana/provisioning/alerting')
entrypoint = File.read(File.join(root, 'common/scripts/grafana-provisioning-entrypoint.sh'))
paths = Dir[File.join(alert_dir, '*.yml')].sort

documents = paths.to_h { |path| [path, YAML.load_file(path)] }
rules = documents.values.flat_map do |document|
  document.fetch('groups', []).flat_map { |group| group.fetch('rules', []) }
end
uids = rules.map { |rule| rule.fetch('uid') }
uid_counts = Hash.new(0)
uids.each { |uid| uid_counts[uid] += 1 }
duplicates = uid_counts.select { |_uid, count| count > 1 }.keys
raise "duplicate alert UIDs: #{duplicates.join(', ')}" unless duplicates.empty?

paths.each do |path|
  name = File.basename(path)
  source = %Q{${SRC_DIR}/alerting/#{name}}
  destination = %Q{${DST_DIR}/alerting/#{name}}
  raise "#{name} is not required by provisioning" unless entrypoint.include?(%Q{[ ! -f "#{source}" ]})
  raise "#{name} is not copied by provisioning" unless entrypoint.include?(%Q{cp "#{source}" "#{destination}"})
end

retired = %w[
  clob_source_timestamp_lag_flapping
  clob_abrupt_upstream_disconnect
  clob_silent_watchdog_disconnect
]
deleted = documents.values.flat_map { |document| document.fetch('deleteRules', []) }
  .map { |rule| rule.fetch('uid') }
missing_deletions = retired - deleted
unless missing_deletions.empty?
  raise "retired alert UIDs lack deleteRules entries: #{missing_deletions.join(', ')}"
end

rules.reject { |rule| rule['isPaused'] }.each do |rule|
  uid = rule.fetch('uid')
  raise "#{uid} lacks noDataState" unless rule.key?('noDataState')
  raise "#{uid} lacks execErrState" unless rule.key?('execErrState')
  raise "#{uid} lacks evaluation duration" unless rule.key?('for')
  %w[severity service family environment].each do |label|
    raise "#{uid} lacks #{label} label" if rule.dig('labels', label).to_s.empty?
  end
  raise "#{uid} lacks an alert summary" if rule.dig('annotations', 'summary').to_s.empty?
end

critical = rules.find { |rule| rule['uid'] == 'mdi_critical_5' }
warning = rules.find { |rule| rule['uid'] == 'mdi_chainlink_reference_transport_failures' }
raise 'missing Chainlink integrity alert' unless critical
raise 'missing Chainlink transport warning' unless warning
critical_expression = critical.dig('data', 0, 'model', 'expr')
warning_expression = warning.dig('data', 0, 'model', 'expr')
transport_codes = %w[
  chainlink_reference_http_request
  chainlink_reference_http_retryable_status
  chainlink_reference_read_response
]
transport_codes.each do |code|
  raise "#{code} remains critical" if critical_expression.include?(code)
  raise "#{code} is absent from transport warning" unless warning_expression.include?(code)
end
raise 'Chainlink transport warning must require repeated failures' unless warning_expression.include?('>= 3')
unless warning.dig('labels', 'severity') == 'warning'
  raise 'Chainlink transport warning severity is not warning'
end

owner = rules.find { |rule| rule['uid'] == 'mdp_duplicate_active_owner' }
raise 'missing duplicate-owner alert' unless owner
expected_owner_expression = 'max(count by (product) (ingester_stream_subscribers > 0))'
unless owner.dig('data', 0, 'model', 'expr') == expected_owner_expression
  raise 'duplicate-owner alert does not count distinct worker series'
end

owner_fixture = {
  'evaluation_interval' => '1m',
  'tests' => [
    {
      'name' => 'multiple subscriptions on one worker are one owner',
      'interval' => '1m',
      'input_series' => [
        {'series' => 'ingester_stream_subscribers{instance="worker-a",product="btc"}', 'values' => '2+0x2'}
      ],
      'promql_expr_test' => [{
        'expr' => expected_owner_expression,
        'eval_time' => '1m',
        'exp_samples' => [{'labels' => '{}', 'value' => 1}]
      }]
    },
    {
      'name' => 'two workers subscribing to one product are duplicate owners',
      'interval' => '1m',
      'input_series' => [
        {'series' => 'ingester_stream_subscribers{instance="worker-a",product="btc"}', 'values' => '1+0x2'},
        {'series' => 'ingester_stream_subscribers{instance="worker-b",product="btc"}', 'values' => '1+0x2'}
      ],
      'promql_expr_test' => [{
        'expr' => expected_owner_expression,
        'eval_time' => '1m',
        'exp_samples' => [{'labels' => '{}', 'value' => 2}]
      }]
    }
  ]
}
output, status = Open3.capture2e(
  'docker', 'exec', '-i', 'prometheus', 'promtool', 'test', 'rules', '/dev/stdin',
  stdin_data: YAML.dump(owner_fixture)
)
puts output
abort 'duplicate-owner Prometheus expression tests failed' unless status.success?

puts "PASS #{rules.size} Grafana alert rules across #{paths.size} provisioned files"
