"""Portable JSON configuration; expand explicit environment references only."""
import json
import os
import re
from pathlib import Path

_PATTERN = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')

def expand(value):
    if isinstance(value, str):
        def replace(match):
            name = match.group(1)
            if name not in os.environ or not os.environ[name]:
                raise ValueError('Missing environment variable: '+name)
            return os.environ[name]
        return _PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [expand(x) for x in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value

def load_config(path):
    value = expand(json.loads(Path(path).read_text()))
    if not isinstance(value, dict):
        raise ValueError('Configuration must be an object')
    return value

def validate_episode(config):
    allowed = {'mode', 'output', 'task', 'limits', 'policy', 'backend', 'preflight'}
    if set(config)-allowed:
        raise ValueError('Unknown episode fields: '+', '.join(sorted(set(config)-allowed)))
    if config.get('mode', 'continuous_stage') != 'continuous_stage':
        raise ValueError('Only continuous_stage is supported')
    for name in ('task', 'limits', 'policy', 'backend', 'output'):
        if name not in config:
            raise ValueError('Missing episode field: '+name)
    if config['limits'] != {'max_rounds': 25, 'max_steps': 4000}:
        raise ValueError('Stage protocol requires 25 rounds / 4000 steps')
    if config['policy'].get('session_mode', 'continuous') != 'continuous':
        raise ValueError('Task requires a continuous policy session')
    if config['policy'].get('decision_unit', 'stage') != 'stage':
        raise ValueError('Only stage decisions are supported')
    cmd = config['backend'].get('command')
    if not isinstance(cmd, list) or not cmd or not all(isinstance(x,str) and x for x in cmd):
        raise ValueError('Backend command must be a nonempty argv list')
    return config
