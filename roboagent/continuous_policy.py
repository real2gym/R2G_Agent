"""Stage fixed outer loop, one fresh persistent CLI context per episode.

The single MCP tool is a blocking decision/result transport, not an agent-selectable
simulator API. Only the outer runner can execute code, advance state or score.
"""
import base64
import hashlib
import json
import os
import queue
import secrets
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from .command import build_command
from .io import write_json
from .policy import CodexPolicy, validate_decision

class ContinuousPolicy(CodexPolicy):
    def __init__(self, config, task, work):
        super().__init__(config, task, work)
        self.base = self.base.replace('The policy has no tools.',
            'The only tool is submit_decision, a transport for your decision and the outer loop feedback; no other tools are allowed.')
        self.base = self.base.replace('after a short test pull', 'after a short grasp or manipulation test').replace(
            'Select the middle drawer handle by image evidence, not simply highest confidence when multiple handles are returned.',
            'Select the requested object using current image evidence; confidence alone is insufficient.')
        self.pending = queue.Queue()
        self.reply = None
        self.proc = None
        self.server = None
        self.server_thread = None
        self.reader = None
        self.events = []
        self.violations = []
        self.closed = False
        self.submissions = 0
        self.error = None
        self.thread_id = None
        self.usage = None
        self.response_ids = set()

    def summary(self):
        return {'mode':'continuous_stage','cli_invocations':int(self.proc is not None),
            'decision_submissions':self.submissions,'thread_id':self.thread_id,
            'model_requests_observed':len(self.response_ids) or None,
            'model_request_count_complete':False,'usage':self.usage,
            'violations':self.violations,'error':self.error}

    def _read(self):
        try:
            with (self.work/'cli.events.jsonl').open('w') as log:
                for line in self.proc.stdout:
                    log.write(line);log.flush()
                    e=json.loads(line);self.events.append(e)
                    if e.get('type')=='thread.started':self.thread_id=e.get('thread_id')
                    if e.get('type')=='turn.completed':self.usage=e.get('usage')
                    if e.get('type')=='response.completed' and e.get('response',{}).get('id'):
                        self.response_ids.add(e['response']['id'])
                    item=e.get('item',{});kind=item.get('type')
                    if kind in ('command_execution','web_search','file_change') or (kind=='mcp_tool_call' and
                        (item.get('server')!='simulation' or item.get('tool')!='submit_decision')):
                        self.violations.append(kind);self.error='Unexpected policy tool';self.proc.terminate()
                    write_json(self.work/'session.json',self.summary())
        except Exception as exc:
            self.error=str(exc)
        finally:
            self.pending.put(None)

    def _start(self, prompt, images):
        owner=self;token=secrets.token_urlsafe(32)
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*a):pass
            def do_POST(self):
                if self.path!='/tool' or not secrets.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                    self.send_error(403);return
                try:
                    size=int(self.headers.get('Content-Length','0'))
                    if not 0<size<=100000:raise ValueError('Invalid request size')
                    req=json.loads(self.rfile.read(size))
                    if owner.closed or req.get('name')!='submit_decision':raise ValueError('Decision transport closed or invalid tool')
                    decision=validate_decision(req.get('arguments',{}))
                    if any(len(v)>18000 for v in decision.values() if isinstance(v,str)):raise ValueError('Decision too long')
                    response=queue.Queue(maxsize=1)
                    owner.pending.put((decision,response))
                    result=response.get(timeout=350)
                except Exception as exc:
                    result={'content':[{'type':'text','text':str(exc)}],'isError':True}
                payload=json.dumps(result).encode();self.send_response(200)
                self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers()
                try:self.wfile.write(payload)
                except BrokenPipeError:pass
        self.server=HTTPServer(('127.0.0.1',0),Handler)
        self.server_thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.server_thread.start()
        cmd=build_command(self.config,self.cwd,'http://127.0.0.1:%d/tool'%self.server.server_port,token,self.work/'final.txt')
        for i,arg in enumerate(cmd):
            if arg.startswith('mcp_servers.simulation.args='):
                cmd[i]='mcp_servers.simulation.args='+json.dumps([str(Path(__file__).with_name('decision_mcp.py')),'http://127.0.0.1:%d/tool'%self.server.server_port,token])
            elif arg.startswith('mcp_servers.simulation.enabled_tools='):
                cmd[i]='mcp_servers.simulation.enabled_tools=["submit_decision"]'
        cmd=cmd[:-1]
        for view in ['agentview','wrist']:cmd+=['-i',str(images[view])]
        cmd+=['-']
        env={k:v for k,v in os.environ.items() if not k.startswith(('CODEX_THREAD','CODEX_SESSION','CODEX_PARENT'))}
        self.stderr=(self.work/'cli.stderr').open('w')
        self.proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,env=env)
        self.proc.stdin.write(prompt);self.proc.stdin.close()
        self.reader=threading.Thread(target=self._read,daemon=True);self.reader.start()

    def decide(self,state,history,images,index):
        start=time.monotonic();prefix=self.work/f'round_{index:03d}'
        if index==0:
            if history:raise ValueError('Continuous task must start with empty history')
            prompt=self.prompt(state,[])+('\nTRANSPORT: Submit every decision with submit_decision, including the final stop decision. '
                'Do not execute anything yourself. Each result contains actual feedback, current self-state and fresh agentview/wrist RGB. '
                'Continue in this same episode context. When the outer framework says episode_ended, give only a brief final acknowledgement and no more tool calls.')
        else:
            if self.reply is None or not history:raise RuntimeError('Missing pending outer-loop feedback')
            prompt='CURRENT ROBOT SELF-STATE:\n'+json.dumps(state,ensure_ascii=False)+'\nPREVIOUS EXECUTION FEEDBACK:\n'+json.dumps(history[-1]['feedback'],ensure_ascii=False)
        Path(str(prefix)+'.prompt.txt').write_text(prompt)
        if index==0:self._start(prompt,images)
        else:
            blocks=[{'type':'text','text':prompt}]
            for view in ['agentview','wrist']:
                blocks += [{'type':'text','text':view},{'type':'image','mimeType':'image/png','data':base64.b64encode(images[view].read_bytes()).decode()}]
            self.reply.put({'content':blocks,'isError':False});self.reply=None
        try:value=self.pending.get(timeout=self.config.get('timeout_seconds',360))
        except queue.Empty:raise TimeoutError('Continuous policy decision timeout')
        if value is None or self.error or self.violations:raise RuntimeError(self.error or 'CLI ended before submitting a decision')
        decision,self.reply=value;self.submissions+=1
        audit={'model':self.config['model'],'effort':self.config['effort'],'elapsed_seconds':time.monotonic()-start,
            'history_rounds':len(history),'history_transmission':'initial instructions, then latest feedback only',
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'images':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in images.items()},'tool_calls':[],
            'allowed_transport':'submit_decision','session':self.summary()}
        write_json(str(prefix)+'.decision.json',decision);write_json(str(prefix)+'.audit.json',audit)
        return decision,audit

    def close(self,reason='outer_loop_ended'):
        self.closed=True
        if self.reply is not None:
            self.reply.put({'content':[{'type':'text','text':json.dumps({'episode_ended':True,'reason':reason,'native_result':'withheld'})}],'isError':False});self.reply=None
        if self.proc:
            try:self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.error = self.error or 'CLI did not finish after episode termination'
                self.proc.terminate()
                try:self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
            self.reader.join(timeout=5);self.proc.stdout.close();self.stderr.close()
            if self.proc.returncode:
                self.error = self.error or 'CLI exited unsuccessfully'
        if self.server:
            self.server.shutdown();self.server.server_close();self.server_thread.join(timeout=5)
        write_json(self.work/'session.json',self.summary())
