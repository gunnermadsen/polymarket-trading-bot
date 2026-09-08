"""Prepare a disabled paper process from a validated UMR catalog and existing playbook.

No API writes, new runtime configuration, or model-specific branches are performed.
"""
import argparse
import copy
import json
from pathlib import Path


def prepare(catalog, template, model_key, process_key, name, preregistration_sha256):
    if len(preregistration_sha256) != 64 or any(c not in '0123456789abcdef' for c in preregistration_sha256):
        raise ValueError('Provide the reviewed preregistration SHA-256 for this process')
    matches = [m for m in catalog['models'] if m['model_key'] == model_key]
    if len(matches) != 1 or not matches[0]['compatible']:
        raise ValueError('Select one compatible catalog model')
    model = matches[0]
    contract = model['contract']
    if not model.get('policy') or not contract.get('qualified_trade_size'):
        raise ValueError('This capability requires its established legacy process procedure')
    result = copy.deepcopy(template)
    control = result['config']['raw']['btc_realtime_paper']
    strategy = control['strategy']
    products = set(model['supported_products'])
    required = [v for v in contract['inputs'] if v['required']]
    if any(v['product'] not in products for v in required):
        raise ValueError('Required source capability is unavailable')
    source_keys = {s if isinstance(s, str) else s['key'] for s in control['sources']}
    if any(v['product'] not in source_keys for v in required):
        raise ValueError('The playbook must select all required shared streams')
    bindings = [dict(slot=v['slot'], product=v['product'], semantics=v['semantics'])
                for v in contract['inputs'] if v['product'] in products and v['product'] in source_keys]
    result.update(name=name, process_key=process_key, process_scope='realtime_paper', enabled=False, status='created')
    for key in ('process_id', 'id', 'created_at', 'updated_at', 'active_run_id'):
        result.pop(key, None)
    result['config']['execution'].update(mode='paper', execute_signals=True, live_capital=False)
    strategy['decision_strategy'] = dict(type='btc_directional_model', **model['selection'])
    strategy.pop('feature_schema_version', None)
    strategy.pop('strategy_version', None)
    strategy['target_size'] = str(contract['qualified_trade_size'])
    strategy['min_seconds_after_open'] = model['schedule']['minimum_seconds_after_open']
    strategy['min_seconds_before_close'] = 300 - model['schedule']['maximum_seconds_after_open']
    strategy['unified_model'] = dict(version=contract['version'], sources=bindings, policy=model['policy'])
    control['preregistration_sha256'] = preregistration_sha256
    control['next_experiment_key'] = process_key + '-run'  # Existing run-key alias, not experiment ownership.
    metadata = result.setdefault('metadata', {})
    # Remove source-specific provenance from the previous template. The new package owns it.
    for key in ('source_training_model_sha256', 'frozen_candidate', 'optional_canonical_candles'):
        metadata.pop(key, None)
    metadata.update(model_key=model_key, model_artifact_sha256=model['selection']['artifact_sha256'],
                    model_feature_schema_version=model['feature_schema_version'],
                    model_feature_schema_sha256=model['selection']['feature_schema_sha256'],
                    umr_contract=contract['version'], deployment_scope='paper_only', production_qualified=False,
                    unavailable_optional_inputs=[v['slot'] for v in contract['inputs'] if not v['required'] and not any(b['slot']==v['slot'] for b in bindings)])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('catalog', 'template', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    for flag in ('model-key', 'process-key', 'name', 'preregistration-sha256'):
        parser.add_argument('--'+flag, required=True)
    args = parser.parse_args()
    result = prepare(json.loads(args.catalog.read_text()), json.loads(args.template.read_text()),
                     args.model_key, args.process_key, args.name, args.preregistration_sha256)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print('Prepared disabled paper process; use the existing start-preview and process APIs to activate.')


if __name__ == '__main__':
    main()
