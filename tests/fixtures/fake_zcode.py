#!/usr/bin/python3
"""Strict subset of the Zcode 0.16.9 stdio contract used by the adapter."""
import argparse
import json
import os
import pathlib
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('command', choices=['app-server'])
p.add_argument('--cwd', required=True)
a = p.parse_args()
assert pathlib.Path.cwd() == pathlib.Path(a.cwd)
assert 'ZCODE_MODEL' not in os.environ
home = pathlib.Path.home()
config = json.loads((home / '.zcode/cli/config.json').read_text())
assert set(config['provider']) == {'test'}
assert 'other' not in json.dumps(config)
assert not (home / '.zcode/cli/history.json').exists()
native = json.loads((home / '.zcode/v2/provider_config.json').read_text())
assert len(native['config']['providerConfigRules']['providerRules']) == 1
assert native['config']['providerConfigRules']['providerRules'][0]['config']['access']['apiKey']
session = 'sess_test'
state = pathlib.Path(os.environ['ZCODE_STORAGE_DIR']) / 'test-session'
provider, model = config['model']['main'].split('/', 1)
effort = None
seq = 0

def write(value):
    print(json.dumps(value), flush=True)

def snapshot():
    return {'session': {'sessionId': session, 'model': {'providerId': provider, 'modelId': model}},
            'settings': {'mode': {'current': 'yolo'}, 'model': {'current': {'providerId': provider, 'modelId': model, 'options': {'reasoningLevel': effort}}},
                         'thoughtLevel': {'current': effort, 'available': [{'value': level} for level in ['low', 'high', 'max']]}},
            'runtime': {'eventSeq': seq}, 'projection': {'status': 'idle'}}

def event(kind, payload, turn='turn_current'):
    global seq
    seq += 1
    write({'method': 'session/event', 'params': {'type': kind, 'sessionId': session, 'turnId': turn, 'seq': seq, 'payload': payload}})

for line in sys.stdin:
    req = json.loads(line)
    assert set(req) == {'id', 'method', 'params'}
    method, params = req['method'], req['params']
    if method == 'v4/command':
        assert set(params) == {'commandId', 'clientId', 'sessionId', 'type', 'payload', 'issuedAt'}
        payload = params['payload']
        kind = params['type']
        if kind == 'createSession':
            assert params['sessionId'] is None
            assert set(payload) == {'workspaceId', 'config', 'mcpServers'}
            assert payload['workspaceId'] == a.cwd
            effort = payload['config'].get('thought', 'max')
            state.write_text(json.dumps({'session': session, 'effort': effort}))
            result = {'type': 'createSession', 'sessionId': session}
        elif kind == 'sendText':
            assert params['sessionId'] == session
            assert set(payload) == {'text', 'requestedDelivery', 'modelSelection', 'mode'}
            assert payload['requestedDelivery'] == 'startNow'
            assert payload['mode'] == 'yolo'
            result = {'type': 'inputAccepted', 'delivery': 'startNow', 'inputId': params['commandId']}
        else:
            raise AssertionError(kind)
        write({'id': req['id'], 'result': {'commandId': params['commandId'], 'status': 'accepted', 'revisionAtDecision': 0, 'result': result}})
        if kind == 'sendText':
            prompt = payload['text']
            if prompt == 'timeout':
                time.sleep(10)
            if prompt == 'truncated':
                raise SystemExit(0)
            if prompt == 'wrong-session':
                session = 'sess_wrong'
            event('turn.started', {})
            usage = {'inputTokens':120,'cacheReadTokens':30,'outputTokens':20,'reasoningTokens':7}
            event('session.updated', {'type':'model_request_completed', 'requestId':'request-one','providerId':provider,'modelId':model,'usage':usage})
            event('turn.completed', {'inputId':params['commandId'], 'resultType':'success','response':'answer:'+prompt,'usage':{'source':'provider', **usage}})
    elif method == 'session/resume':
        assert set(params) == {'sessionId', 'mcpServers'}
        stored = json.loads(state.read_text())
        assert params['sessionId'] == stored['session']
        session, effort = stored['session'], stored['effort']
        write({'id': req['id'], 'result': snapshot()})
    elif method == 'session/read':
        assert params == {'sessionId': session}
        write({'id': req['id'], 'result': snapshot()})
    elif method == 'session/subscribe':
        assert params == {'sessionId': session, 'deliveryKind': 'desktop-continuous', 'afterSeq': seq, 'includeSnapshot': False}
        write({'id': req['id'], 'result': {'sessionId': session, 'eventSeq': seq, 'events': []}})
    else:
        raise AssertionError(method)
