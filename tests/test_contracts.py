import base64
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from roboagent.config import expand, load_config, validate_episode
from roboagent.policy import TEXT_FIELDS, validate_decision, CodexPolicy
from roboagent.boundary import check_code, split_observation, validate_feedback
from roboagent.perception import PerceptionClient, np_b64, b64_np
from fixtures import STATE, PNG

class Contracts(unittest.TestCase):
    def test_decision_exact_shape_and_stop(self):
        d=dict.fromkeys(TEXT_FIELDS,'');d.update(stop=False,code='print(1)')
        self.assertEqual(validate_decision(d),d)
        for invalid in [dict(d,extra=[]),dict(d,stop=1),dict(d,code=''),dict(d,stop=True)]:
            with self.assertRaises(ValueError):validate_decision(invalid)
        d.update(stop=True,code='',stop_reason='visual stop');validate_decision(d)

    def test_source_and_schema_have_no_removed_feature(self):
        root=Path(__file__).resolve().parents[1]/'roboagent'
        forbidden='ski'+'ll'
        for p in root.rglob('*'):
            if p.suffix in ('.py','.txt','.json','.html'):
                self.assertNotIn(forbidden,p.read_text().lower(),str(p))
        schema=json.loads((root/'resources/decision.schema.json').read_text())
        self.assertEqual(set(schema['required']),set(TEXT_FIELDS)|{'stop'})

    def test_code_boundary(self):
        check_code('import numpy as np\nx=np.array([1.,2.,3.])\nprint(x.mean())')
        for code in ['import os','open("x")','np.load("x")','x.__class__','getattr(x,"value")']:
            with self.assertRaises(ValueError):check_code(code)

    def test_policy_input_excludes_native_result(self):
        obs=dict(STATE,images={'agentview':PNG,'wrist':PNG});split_observation(obs)
        with self.assertRaises(ValueError):split_observation(dict(obs,native_success=True))
        fb=dict(stdout='feedback',stderr='',sandbox_rc=0,before=STATE,after=STATE)
        validate_feedback(fb)
        with self.assertRaises(ValueError):validate_feedback(dict(fb,reward=1))

    def test_explicit_env_and_missing_value(self):
        with patch.dict(os.environ,{'R2G_TEST_PATH':'a path with spaces'}):
            self.assertEqual(expand({'command':['${R2G_TEST_PATH}']}),{'command':['a path with spaces']})
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'Missing environment'):expand('${R2G_MISSING}')

    def test_numpy_serialization_rejects_pickle(self):
        arr=np.arange(12,dtype=np.float32).reshape(3,4)
        np.testing.assert_array_equal(b64_np(np_b64(arr)),arr)
        stream=io.BytesIO();np.save(stream,np.array([{}],dtype=object))
        with self.assertRaises(ValueError):b64_np(base64.b64encode(stream.getvalue()).decode())

    def test_real_service_contract_requests(self):
        client=PerceptionClient('http://sam.invalid','http://grasp.invalid')
        depth=np.ones((2,3));mask=np.array([[0,1,0],[1,1,0]],np.uint8)
        calls=[]
        def urlopen(req,timeout):
            calls.append((req.full_url,json.loads(req.data)))
            if req.full_url.endswith('/segment'):
                result={'results':[{'mask_base64':np_b64(mask),'shape':[2,3],'score':.9}]}
            else:result={'grasps_base64':np_b64(np.eye(4)[None]),'scores_base64':np_b64([.8])}
            return io.BytesIO(json.dumps(result).encode())
        with patch('urllib.request.urlopen',side_effect=urlopen):
            out=client.segment_sam3_text_prompt(np.zeros((2,3,3),np.uint8),'target')
            poses,scores=client.plan_grasp(depth,np.eye(3),out[0]['mask'])
        self.assertEqual(calls[0][0],'http://sam.invalid/segment')
        self.assertEqual(calls[1][0],'http://grasp.invalid/plan')
        np.testing.assert_array_equal(b64_np(calls[1][1]['depth_base64']),depth)
        self.assertEqual(poses.shape,(1,4,4));self.assertEqual(scores.shape,(1,))

    def test_service_failures_are_not_fabricated(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(ValueError):PerceptionClient()
        client=PerceptionClient('http://sam.invalid','http://grasp.invalid')
        with patch('urllib.request.urlopen',side_effect=ConnectionError('offline')):
            with self.assertRaises(ConnectionError):client.plan_grasp(np.ones((2,2)),np.eye(3),np.ones((2,2)))

    def test_episode_rejects_unrecognized_options(self):
        root=Path(__file__).resolve().parents[1]
        c=json.loads((root/'examples/episode.json').read_text())
        validate_episode(c)
        for changed in [dict(c,obsolete_input=[]),dict(c,mode='other'),dict(c,limits={'max_rounds':1,'max_steps':1})]:
            with self.assertRaises(ValueError):validate_episode(changed)

    def test_stage_only_and_fresh_session_rejected(self):
        root=Path(__file__).resolve().parents[1]
        c=json.loads((root/'examples/episode.json').read_text())
        for field,value in [('decision_unit','action'),('session_mode','fresh')]:
            changed=dict(c,policy=dict(c['policy'],**{field:value}))
            with self.assertRaises(ValueError):validate_episode(changed)

    def test_point_service_masks(self):
        client=PerceptionClient('http://sam.invalid','http://grasp.invalid')
        arr=np.array([[[0,1],[1,0]]],dtype=np.uint8)
        payload={'masks_base64':base64.b64encode(arr.tobytes()).decode(),'masks_dtype':'uint8','masks_shape':[1,2,2],'scores':[.95]}
        with patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(payload).encode())) as call:
            result=client.segment_sam3_point_prompt(np.zeros((2,2,3),np.uint8),(1,0))
        self.assertTrue(result[0]['mask'][0,1])
        self.assertEqual(json.loads(call.call_args.args[0].data)['point_coords'],[1,0])
