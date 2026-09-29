"""Shared prompt and decision validation; execution lives in ContinuousPolicy."""
import json
from pathlib import Path

RESOURCES = Path(__file__).parent / 'resources'
TEXT_FIELDS = ['visual_observation', 'inference', 'uncertainty', 'failure_hypothesis',
    'local_correction', 'code', 'expected_visual_effect', 'stop_reason']


def validate_decision(value):
    if not isinstance(value, dict):
        raise ValueError('Decision must be an object')
    if set(value) != set(TEXT_FIELDS) | {'stop'}:
        raise ValueError('Invalid decision keys')
    if any(not isinstance(value[k], str) for k in TEXT_FIELDS) or type(value['stop']) is not bool:
        raise ValueError('Invalid decision types')
    if value['stop'] and (value['code'].strip() or not value['stop_reason'].strip()):
        raise ValueError('Stop needs empty code and an explicit reason')
    if not value['stop'] and not value['code'].strip():
        raise ValueError('Action decision needs code')
    return value


class CodexPolicy:
    def __init__(self, config, task, work):
        self.config, self.task = config, task
        self.work = Path(work)
        self.cwd = self.work / 'empty_policy_cwd'
        self.cwd.mkdir()
        self.base = Path(config.get('prompt_path', RESOURCES / 'ego_policy.txt')).read_text(encoding='utf-8')
        unit = config.get('decision_unit', 'stage')
        if unit not in ('action', 'stage'):
            raise ValueError('decision_unit must be action or stage')
        if unit == 'stage':
            self.base += '\n\n' + (RESOURCES / 'stage_decision.txt').read_text(encoding='utf-8')

    def prompt(self, state, history):
        return (self.base + '\nTASK LANGUAGE:\n' + self.task +
            '\nCURRENT ROBOT SELF-STATE:\n' + json.dumps(state, ensure_ascii=False) +
            '\nTHIS EPISODE HISTORY:\n' + json.dumps(history, ensure_ascii=False))
