"""Scene routing and shared input contract; no simulator imports or side effects."""
from pathlib import Path
import re
from roboagent.config import load_config

ADAPTERS = {
    **{f'D{i:02}': 'roboagent.backends.droid' for i in range(1, 9)},
    **{f'D{i:02}': 'roboagent.backends.batch_backend' for i in range(9, 13)},
    **{f'E{i:02}': 'roboagent.backends.batch_backend' for i in range(1, 3)},
    **{f'E{i:02}': 'roboagent.backends.ego_backend' for i in range(3, 13)},
}


def adapter_for(scene_id):
    if not isinstance(scene_id, str) or scene_id not in ADAPTERS:
        raise ValueError('scene_id must be an explicit D01–D12 or E01–E12 identifier')
    return ADAPTERS[scene_id]


def normalize_config(value, base):
    """Validate public fields and translate the scene ID for native profiles.

    task_id is an optional matching legacy alias, never a routing fallback.
    Relative filesystem paths are resolved against the configuration directory.
    No assets, services or output directories are opened here.
    """
    c = dict(value)
    allowed = {'scene_id', 'task_id', 'scene_dir', 'artifact_dir', 'language',
               'max_steps', 'success_criteria_path', 'success_criteria_sha256'}
    if set(c) - allowed:
        raise ValueError('Unknown backend fields: '+', '.join(sorted(set(c)-allowed)))
    adapter_for(c.get('scene_id'))
    if 'task_id' in c and c['task_id'] != c['scene_id']:
        raise ValueError('task_id conflicts with scene_id')
    c['task_id'] = c['scene_id']  # Existing profile and ready-message contract.
    for key in ('scene_dir', 'artifact_dir', 'language'):
        if not isinstance(c.get(key), str) or not c[key].strip():
            raise ValueError('Backend requires nonempty '+key)
    c.setdefault('max_steps', 4000)
    if type(c['max_steps']) is not int or c['max_steps'] != 4000:
        raise ValueError('Stage backend requires max_steps=4000')
    criteria = ('success_criteria_path', 'success_criteria_sha256')
    if c['scene_id'] == 'D01':
        if any(key in c for key in criteria):
            raise ValueError('D01 uses its retained inline criterion; omit external criterion fields')
    else:
        if not isinstance(c.get(criteria[0]), str) or not c[criteria[0]].strip():
            raise ValueError('Backend requires success_criteria_path')
        if not isinstance(c.get(criteria[1]), str) or not re.fullmatch(r'[0-9a-fA-F]{64}', c[criteria[1]]):
            raise ValueError('Backend requires a 64-digit success_criteria_sha256')
        c[criteria[1]] = c[criteria[1]].lower()
    base = Path(base).resolve()
    for key in ('scene_dir', 'artifact_dir', 'success_criteria_path'):
        if key in c:
            path = Path(c[key]).expanduser()
            c[key] = str((path if path.is_absolute() else base/path).resolve())
    return c


def load_backend_config(path):
    path = Path(path).resolve()
    return normalize_config(load_config(path), path.parent)
