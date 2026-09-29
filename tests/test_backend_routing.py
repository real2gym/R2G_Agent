"""Unified entry-point tests; never import MuJoCo or launch model services."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from roboagent.backends.__main__ import main
from roboagent.backends.routing import adapter_for, load_backend_config, normalize_config


def config(scene='E03'):
    c={'scene_id':scene,'scene_dir':'scene','artifact_dir':'run','language':'task','max_steps':4000}
    if scene!='D01':
        c.update(success_criteria_path='criteria.json',success_criteria_sha256='a'*64)
    return c


class BackendRouting(unittest.TestCase):
    def test_all_24_routes(self):
        for i in range(1,13):
            for family in ['D','E']:
                scene=f'{family}{i:02}'
                expected=('droid' if family=='D' and i<=8 else
                          'batch_backend' if family=='D' or i<=2 else 'ego_backend')
                self.assertEqual(adapter_for(scene),'roboagent.backends.'+expected)

    def test_explicit_identifier_no_default_or_guess(self):
        for scene in [None,'','D1','d01','D00','D13','E13',[],1]:
            with self.subTest(scene=scene),self.assertRaises(ValueError):adapter_for(scene)
        c=config();c['task_id']=c.pop('scene_id')
        with self.assertRaisesRegex(ValueError,'scene_id'):normalize_config(c,'.')

    def test_alias_conflict_is_rejected(self):
        c=config();c['task_id']='E04'
        with self.assertRaisesRegex(ValueError,'conflicts'):normalize_config(c,'.')
        c['task_id']='E03';self.assertEqual(normalize_config(c,'.')['task_id'],'E03')

    def test_relative_paths_and_environment(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'R2G_ROUTING_SCENE':'E12'}):
            p=Path(tmp)/'backend.json';c=config();c['scene_id']='${R2G_ROUTING_SCENE}';p.write_text(json.dumps(c))
            result=load_backend_config(p)
            self.assertEqual(result['scene_id'],'E12')
            self.assertEqual(result['scene_dir'],str(Path(tmp).resolve()/'scene'))
            self.assertEqual(result['artifact_dir'],str(Path(tmp).resolve()/'run'))
            self.assertFalse((Path(tmp)/'run').exists())

    def test_common_fields_and_budget(self):
        for field in ['scene_dir','artifact_dir','language']:
            for value in [None,'',123]:
                c=config();c[field]=value
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):normalize_config(c,'.')
        for budget in [True,'4000',1]:
            c=config();c['max_steps']=budget
            with self.assertRaises(ValueError):normalize_config(c,'.')
        c=config();c.pop('max_steps');original=copy.deepcopy(c)
        self.assertEqual(normalize_config(c,'.')['max_steps'],4000);self.assertEqual(c,original)
        with self.assertRaisesRegex(ValueError,'Unknown'):normalize_config(dict(c,adapter='wrong'),'.')

    def test_criteria_requirements_and_d01_exception(self):
        self.assertNotIn('success_criteria_path',normalize_config(config('D01'),'.'))
        c=config('D01');c['success_criteria_path']='ignored.json'
        with self.assertRaisesRegex(ValueError,'inline'):normalize_config(c,'.')
        for key in ['success_criteria_path','success_criteria_sha256']:
            c=config('D02');c.pop(key)
            with self.assertRaises(ValueError):normalize_config(c,'.')
        c=config();c['success_criteria_sha256']='bad'
        with self.assertRaises(ValueError):normalize_config(c,'.')

    def test_dispatch_uses_validated_config_once_and_no_stdout(self):
        import io
        for scene in ['D01','D08','D09','E01','E02','E03','E12']:
            with self.subTest(scene=scene),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'config.json';p.write_text(json.dumps(config(scene)))
                with patch('roboagent.backends.__main__.runpy.run_module') as run,patch('sys.stdout',new_callable=io.StringIO) as output:
                    main([str(p)])
                run.assert_called_once();self.assertEqual(run.call_args.args[0],adapter_for(scene))
                passed=run.call_args.kwargs['init_globals']['BACKEND_CONFIG']
                self.assertEqual(passed['task_id'],scene);self.assertEqual(passed['scene_id'],scene)
                self.assertEqual(run.call_args.kwargs['run_name'],'__main__');self.assertEqual(output.getvalue(),'')

    def test_invalid_input_never_dispatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json';p.write_text(json.dumps(config('E99')))
            with patch('roboagent.backends.__main__.runpy.run_module') as run:
                with self.assertRaises(ValueError):main([str(p)])
                run.assert_not_called()

    def test_public_help_and_example_command(self):
        result=subprocess.run([sys.executable,'-m','roboagent.backends','--help'],capture_output=True,text=True,check=True)
        self.assertIn('scene_id',result.stdout)
        root=Path(__file__).resolve().parents[1]
        episode=json.loads((root/'examples/episode.json').read_text())
        self.assertEqual(episode['backend']['command'][1:3],['-m','roboagent.backends'])
        self.assertEqual(episode['task']['task_id'],'${R2G_SCENE_ID}')
        self.assertEqual(json.loads((root/'examples/backend.json').read_text())['scene_id'],'${R2G_SCENE_ID}')
