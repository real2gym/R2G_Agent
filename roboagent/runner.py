"""Observe → decide/code → guarded execution → feedback → independent outcome."""
import datetime
import json
import shutil
import subprocess
from pathlib import Path
from .boundary import split_observation, validate_feedback
from .io import sha256, write_json
from .transport import Backend

ARTIFACTS = ('agentview.mp4', 'wrist.mp4', 'native_result.json', 'framework_manifest.json')


def collect_artifacts(source, destination):
    destination = Path(destination)
    for name in ARTIFACTS:
        target = destination/name
        if source['kind'] == 'local':
            shutil.copyfile(Path(source['path'])/name, target)
        elif source['kind'] == 'scp':
            # The operator supplies the trusted host/path; no model values enter argv.
            remote = source['host'] + ':' + source['path'].rstrip('/') + '/' + name
            subprocess.run(['scp', '-q', '-P', str(source.get('port', 22)), remote, str(target)],
                           check=True, timeout=120)
        else:
            raise ValueError('Unsupported artifact source')
        if not target.stat().st_size:
            raise ValueError('Empty required artifact: ' + name)
    result = json.loads((destination/'native_result.json').read_text())
    if type(result.get('native_success')) is not bool:
        raise ValueError('Missing independent native result')
    return result


def run(config, base):
    from .config import validate_episode
    validate_episode(config)
    config = {**config, 'policy': dict(config.get('policy', {}))}
    if config.get('mode', 'continuous_stage') != 'continuous_stage':
        raise ValueError('Only continuous_stage is supported')
    if config['policy'].get('session_mode', 'continuous') != 'continuous':
        raise ValueError('Each task requires a continuous policy session; per-round fresh mode was removed')
    config.setdefault('mode', 'continuous_stage')
    config['policy']['session_mode'] = 'continuous'
    config['policy'].setdefault('decision_unit', 'stage')
    if config['limits'] != {'max_rounds': 25, 'max_steps': 4000}:
        raise ValueError('Stage protocol uses 25 rounds / 4000 steps; change prompt and backend together for another protocol')
    base = Path(base)
    output = Path(config['output']).expanduser()
    if not output.is_absolute():
        output = base/output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite a prior trial.
    work, public = output/'work', output/'public'
    work.mkdir(); public.mkdir(); (public/'episode').mkdir()
    resources = Path(__file__).parent/'resources'
    shutil.copy(resources/'index.html', public/'index.html')
    write_json(work/'config.json', config)
    from .continuous_policy import ContinuousPolicy
    policy = ContinuousPolicy(config['policy'], config['task']['language'], work)
    data = {'mode': config.get('mode', 'continuous_stage'), 'status': 'initializing', 'task': config['task'], 'rounds': [], 'steps': 0,
            'native': None, 'videos': False, 'started_at': datetime.datetime.now().isoformat()}
    history = []
    backend = None
    ready = False
    reason = 'not_started'
    failure = None

    def save():
        if hasattr(policy, 'summary'):
            data['policy_session'] = policy.summary()
        write_json(public/'run.json', data)

    def observe(value, index):
        state, blobs = split_observation(value)
        name = f'round_{index:03d}_step_{state["executed_steps"]:04d}'
        folder = public/'episode'/name
        folder.mkdir()
        images = {}
        for view, blob in blobs.items():
            images[view] = folder/(view+'.png')
            images[view].write_bytes(blob)
        data.update(latest=name, steps=state['executed_steps'])
        save()
        return state, images, name

    save()
    try:
        backend = Backend(config['backend']['command'], work/'backend.stderr',
                          config['backend'].get('timeout_seconds', 360))
        response = backend.receive()
        if response.get('ready') is not True:
            raise RuntimeError('Backend not ready')
        ready = True
        for key in ('suite', 'task_id', 'init_id', 'seed', 'language'):
            if response.get('task', {}).get(key) != config['task'].get(key):
                raise ValueError('Backend task differs from controller config: '+key)
        if response.get('limits', {}).get('max_steps') != config['limits']['max_steps']:
            raise ValueError('Backend control budget differs from controller config')
        state, images, name = observe(response['observation'], 0)
        data['status'] = 'running'; save()
        for index in range(config['limits']['max_rounds']):
            if state['executed_steps'] >= config['limits']['max_steps']:
                reason = 'control_step_budget'; break
            decision, audit = policy.decide(state, history, images, index)
            entry = {'round': index+1, 'image_dir': name, 'robot_state': state,
                     'decision': decision, 'executed': False, 'feedback': [],
                     'model_seconds': audit['elapsed_seconds']}
            data['rounds'].append(entry); save()
            if decision['stop']:
                reason = decision['stop_reason']; break
            response = backend.request({'op': 'exec', 'code': decision['code']})
            feedback = validate_feedback(response['feedback'])
            entry.update(executed=True, feedback=[feedback])
            state, images, name = observe(response['observation'], index+1)
            history.append({'round': index+1, 'decision': decision, 'feedback': feedback})
            save()
        else:
            reason = 'model_round_budget'
    except Exception as exc:
        reason = type(exc).__name__+': '+str(exc)
        failure = reason
    finally:
        data['stop_reason'] = reason
        if hasattr(policy, 'close'):
            try:
                policy.close(reason)
                if policy.error or policy.violations:
                    failure = failure or policy.error or 'Unexpected policy tool'
            except Exception as exc:
                failure = failure or str(exc)
        if ready:
            try:
                answer = backend.request({'op': 'finish', 'reason': reason})
                if answer.get('finished') is not True:
                    raise RuntimeError('Backend did not finish')
                result = collect_artifacts(config['backend']['artifacts'], public/'episode')
                data['native'] = {k: v for k, v in result.items() if k != 'native_round_records'}
                data['videos'] = True
                write_json(work/'artifact_hashes.json', {n: sha256(public/'episode'/n) for n in ARTIFACTS})
            except Exception as exc:
                data['finalization_error'] = str(exc)
                failure = failure or str(exc)
        if backend:
            backend.close()
        data['status'] = 'error' if failure else 'finished'
        data['finished_at'] = datetime.datetime.now().isoformat()
        save()
    if failure:
        raise RuntimeError(f'{failure}; audit retained at {output}')
    return output
