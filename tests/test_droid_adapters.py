"""Adapter regression tests without launching a renderer or a model service."""
import ast
import builtins
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
from roboagent.boundary import check_code
from roboagent.perception import PerceptionClient

ROOT=Path(__file__).resolve().parents[1]

def function(path,name,namespace):
    tree=ast.parse(path.read_text())
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]

class DroidAdapters(unittest.TestCase):
    def test_all_eight_routed_with_matching_prompt(self):
        source=(ROOT/'roboagent/backends/droid.py').read_text()
        for i in range(1,9):
            self.assertIn(repr(f'D{i:02}'),source)
            prompt='droid_policy.txt' if i==1 else f'droid_d{i:02}_policy.txt'
            self.assertTrue((ROOT/'roboagent/resources'/prompt).is_file())
        self.assertIn("'model.xml' if TASK_ID == 'D02'",source)
        self.assertIn("np.load(ROOT / 'initial_qpos.npy')",source)
        self.assertIn("MODEL.joint('block_free')",source)
        self.assertIn("PROFILE.finish(STEP_COUNT, ARTIFACT_DIR)",source)

    def test_retained_stage_routes_perception_and_persists_variables(self):
        client=PerceptionClient('http://sam.invalid','http://grasp.invalid')
        client.segment_sam3_text_prompt=Mock(return_value=[{'mask':np.ones((2,2),bool),'score':.9}])
        client.plan_grasp=Mock(return_value=(np.eye(4)[None],np.array([.8])))
        ns={'np':np,'DATA':SimpleNamespace(qpos=np.zeros(8)),'PERCEPTION':client,'builtins':builtins,'SCOPE':{},'state':lambda:{'executed_steps':0}}
        run=function(ROOT/'roboagent/backends/droid.py','run_code',ns)
        run("seen=segment_sam3_text_prompt(np.zeros((2,2,3)), 'target')\nposes,scores=plan_grasp(np.ones((2,2)),np.eye(3),seen[0]['mask'])")
        run('pose_count=len(poses)')
        self.assertEqual(ns['SCOPE']['pose_count'],1)
        client.segment_sam3_text_prompt.assert_called_once();client.plan_grasp.assert_called_once()
        with self.assertRaises(ValueError):run('import os')
        with self.assertRaisesRegex(RuntimeError,'Contact feedback'):
            ns['PROFILE']=None;run('get_gripper_contact_feedback()')

    def test_each_criterion_evaluator_rejects_tampering_before_trace_load(self):
        for module in ['droid_evaluator','droid_band','droid_drawer','droid_door','droid_kitchen']:
            with self.subTest(module=module),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'criteria.json';p.write_text('{}')
                ns={'Path':Path,'hashlib':hashlib,'json':json,'np':np}
                evaluate=function(ROOT/'roboagent/backends'/f'{module}.py','evaluate',ns)
                with self.assertRaises((ValueError,AssertionError)):
                    evaluate(Path(tmp)/'missing_trace.npz',p,'wrong-digest')
