"""Single stage-decision MCP transport; no execution or file tools."""
import json
import sys
import urllib.request
from pathlib import Path
if __package__:
    from .policy import validate_decision
else:
    from policy import validate_decision
DEFINITION = json.loads((Path(__file__).parent/'resources/decision.schema.json').read_text())
TOOLS = [{'name': 'submit_decision', 'description': 'Submit one stage decision and wait for actual feedback and fresh RGB. stop=true ends the episode.', 'inputSchema': DEFINITION}]
def validate_call(name, arguments):
    if name != 'submit_decision':
        raise ValueError('Only submit_decision is available')
    validate_decision(arguments)
    if any(len(v)>18000 for v in arguments.values() if isinstance(v,str)):
        raise ValueError('Decision text too long')

def serve(endpoint, token, input_stream=sys.stdin, output_stream=sys.stdout):
    for line in input_stream:
        req = None
        try:
            req = json.loads(line)
            if 'id' not in req:  # initialized/cancelled notifications
                continue
            method = req.get('method')
            if method == 'initialize':
                result = {'protocolVersion': req.get('params', {}).get('protocolVersion', '2024-11-05'),
                          'capabilities': {'tools': {}},
                          'serverInfo': {'name': 'roboagent_simulation', 'version': '0.2.0'}}
            elif method == 'tools/list':
                result = {'tools': TOOLS}
            elif method in ('ping',):
                result = {}
            elif method in ('resources/list', 'resources/templates/list', 'prompts/list'):
                result = { {'resources/list':'resources', 'resources/templates/list':'resourceTemplates', 'prompts/list':'prompts'}[method]: []}
            elif method == 'tools/call':
                p = req['params']
                validate_call(p['name'], p.get('arguments', {}))
                request = urllib.request.Request(endpoint, data=json.dumps(p).encode(),
                    headers={'Authorization': 'Bearer '+token, 'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=360) as response:
                    result = json.load(response)
            else:
                raise ValueError('Unsupported MCP method')
            answer = {'jsonrpc': '2.0', 'id': req['id'], 'result': result}
        except Exception as exc:
            answer = {'jsonrpc': '2.0', 'id': req.get('id') if isinstance(req, dict) else None,
                      'error': {'code': -32602, 'message': type(exc).__name__+': '+str(exc)}}
        output_stream.write(json.dumps(answer)+'\n'); output_stream.flush()


if __name__ == '__main__':
    serve(sys.argv[1], sys.argv[2])
