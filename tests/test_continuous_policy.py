import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from roboagent.runner import run
from roboagent.continuous_policy import ContinuousPolicy
from fixtures import MOCK_BACKEND,TASK,STATE,PNG

MOCK='''import sys,json,urllib.request,pathlib
args=sys.argv[1:];initial=sys.stdin.read()
v=next(x.split('=',1)[1] for x in args if x.startswith('mcp_servers.simulation.args='))
_,url,token=json.loads(v)
def call(d):
 r=urllib.request.Request(url,data=json.dumps({'name':'submit_decision','arguments':d}).encode(),headers={'Authorization':'Bearer '+token})
 return json.load(urllib.request.urlopen(r,timeout=10))
print(json.dumps({'type':'thread.started','thread_id':'single-mock-thread'}),flush=True)
d={k:'' for k in ['visual_observation','inference','uncertainty','failure_hypothesis','local_correction','code','expected_visual_effect','stop_reason']}
d.update(stop=False,code='print(1)')
r=call(d)
assert sum(b['type']=='image' for b in r['content'])==2
assert 'actual action feedback' in r['content'][0]['text']
assert 'native_success' not in str(r)
assert 'THIS EPISODE HISTORY' not in r['content'][0]['text']
d.update(stop=True,code='',stop_reason='mock stop')
r=call(d);assert 'episode_ended' in r['content'][0]['text']
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':100,'cached_input_tokens':50,'output_tokens':20}}),flush=True)
'''

class ContinuousTests(unittest.TestCase):
    def test_fixed_loop_uses_one_session_and_delta_feedback(self, mode='continuous_stage'):
        with tempfile.TemporaryDirectory() as tmp:
            r=pathlib.Path(tmp);cli=r/'cli';cli.write_text('#!'+sys.executable+'\n'+MOCK);cli.chmod(0o700)
            backend=r/'backend.py';backend.write_text(MOCK_BACKEND)
            c={'output':str(r/'run'),'task':TASK,
               'policy':{'cli':str(cli),'model':'gpt-6-astra','effort':'medium','session_mode':'continuous','timeout_seconds':5},
               'limits':{'max_rounds':25,'max_steps':4000},'backend':{'command':[sys.executable,str(backend),str(r/'artifacts'),json.dumps(TASK),json.dumps(STATE),PNG], 'artifacts':{'kind':'local','path':str(r/'artifacts')}}}
            out=run(c,r);d=json.loads((out/'public/run.json').read_text())
            self.assertEqual(d['mode'], mode)
            self.assertEqual([x['executed'] for x in d['rounds']],[True,False])
            self.assertEqual(d['policy_session']['cli_invocations'],1)
            self.assertEqual(d['policy_session']['decision_submissions'],2)
            self.assertEqual(d['policy_session']['thread_id'],'single-mock-thread')
            self.assertEqual(d['policy_session']['usage']['input_tokens'],100)
            self.assertIsNone(d['policy_session']['model_requests_observed'])
            self.assertNotIn('Observation API:',(out/'work/round_001.prompt.txt').read_text())
            self.assertTrue(d['native']['native_success']) # mock only

    def test_cli_exit_without_decision_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=pathlib.Path(tmp);cli=r/'cli';cli.write_text('#!'+sys.executable+'\nimport sys\nsys.stdin.read()\n');cli.chmod(0o700)
            p=ContinuousPolicy({'cli':str(cli),'model':'gpt-6-astra','effort':'medium','timeout_seconds':2},'test',r)
            images={v:r/(v+'.png') for v in ['agentview','wrist']}
            for f in images.values():f.write_bytes(b'fake')
            try:
                with self.assertRaisesRegex(RuntimeError,'ended before'):p.decide(STATE,[],images,0)
            finally:p.close()
            self.assertIsNotNone(p.proc.poll())

    def test_mcp_only_advertises_decision_transport(self):
        script=pathlib.Path(__file__).resolve().parents[1]/'roboagent/decision_mcp.py'
        req='\n'.join(json.dumps(x) for x in [{'id':1,'method':'initialize'},{'id':2,'method':'tools/list'},{'id':3,'method':'tools/call','params':{'name':'observe','arguments':{}}}])+'\n'
        p=subprocess.run([sys.executable,str(script),'http://127.0.0.1:1/tool','fake'],input=req,text=True,capture_output=True,check=True)
        a=[json.loads(x) for x in p.stdout.splitlines()]
        self.assertEqual([t['name'] for t in a[1]['result']['tools']],['submit_decision'])
        self.assertIn('error',a[2])
