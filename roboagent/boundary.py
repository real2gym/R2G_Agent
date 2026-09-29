"""Policy-facing serialization checks and generated-code guard.

Defense in depth for controlled research, NOT a hardened Python sandbox.
"""
import ast
import base64
import math

FORBIDDEN_NAMES = {'env', 'APIS', 'INPUTS', 'open', 'exec', 'eval', 'compile',
    'getattr', 'setattr', 'delattr', 'vars', 'globals', 'locals', 'dir', 'help',
    'input', 'breakpoint', 'type', 'object', 'super', 'memoryview', 'bytearray',
    'bytes', '__import__'}
FORBIDDEN_ATTRS = {'load', 'save', 'savez', 'savez_compressed', 'loadtxt', 'savetxt',
    'fromfile', 'tofile', 'memmap', 'ctypeslib', 'ctypes', 'f2py', 'lib', 'testing',
    'core', 'getbuffer', 'getvalue', 'format', 'format_map', 'dump', 'dumps',
    'loads', 'read', 'write', 'system', 'popen', 'os', 'sys', 'builtins'}


def check_code(code):
    if len(code) > 18000:
        raise ValueError('code_too_large')
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.AsyncFor,
                ast.Await, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith)):
            raise ValueError('unsupported_code_construct')
        if isinstance(node, ast.Import) and any(a.name not in {'numpy', 'math'} for a in node.names):
            raise ValueError('unsupported_import')
        if isinstance(node, ast.ImportFrom) and (node.level or
                node.module != 'scipy.spatial.transform' or any(a.name != 'Rotation' for a in node.names)):
            raise ValueError('unsupported_import')
        if isinstance(node, ast.Name) and (node.id.startswith('_') or node.id in FORBIDDEN_NAMES):
            raise ValueError('forbidden_name')
        if isinstance(node, ast.Attribute) and (node.attr.startswith('_') or node.attr in FORBIDDEN_ATTRS):
            raise ValueError('forbidden_attribute')
    return tree


STATE_KEYS = {'robot_cartesian_pos', 'robot_joint_pos', 'executed_steps'}


def state_only(state):
    if set(state) != STATE_KEYS:
        raise ValueError('Unexpected policy robot-state fields')
    for key in ('robot_cartesian_pos', 'robot_joint_pos'):
        if len(state[key]) != 8 or any(type(v) not in (int, float) or not math.isfinite(v) for v in state[key]):
            raise ValueError('Expected 7 pose/joint values plus gripper fraction')
    if type(state['executed_steps']) is not int or state['executed_steps'] < 0:
        raise ValueError('Invalid control step count')
    return state


def split_observation(observation):
    if set(observation) != STATE_KEYS | {'images'}:
        raise ValueError('Unexpected observation fields; refusing policy input')
    images = observation['images']
    if set(images) != {'agentview', 'wrist'}:
        raise ValueError('Exactly two authorized views required')
    decoded = {name: base64.b64decode(value, validate=True) for name, value in images.items()}
    if any(not blob.startswith(b'\x89PNG\r\n\x1a\n') for blob in decoded.values()):
        raise ValueError('Expected PNG observations')
    return state_only({k: observation[k] for k in STATE_KEYS}), decoded


def validate_feedback(feedback):
    if set(feedback) != {'stdout', 'stderr', 'sandbox_rc', 'before', 'after'}:
        raise ValueError('Unexpected feedback fields; native feedback is forbidden')
    state_only(feedback['before'])
    state_only(feedback['after'])
    if not isinstance(feedback['stdout'], str) or not isinstance(feedback['stderr'], str):
        raise ValueError('Invalid action text feedback')
    if type(feedback['sandbox_rc']) is not int:
        raise ValueError('Invalid code return status')
    return feedback
