"""CLI command for one persistent stage-decision session."""
import json
import sys
from pathlib import Path

def build_command(policy, cwd, endpoint, token, output):
    cmd = [policy.get('cli', 'codex'), 'exec', '--ignore-user-config', '--ignore-rules',
           '--ephemeral', '--skip-git-repo-check', '-C', str(cwd), '-m', policy['model'],
           '--sandbox', 'read-only', '--json', '-o', str(output)]
    config = {'model_reasoning_effort': policy['effort'], 'project_doc_max_bytes': 0,
              'web_search': 'disabled', 'approval_policy': 'never',
              'mcp_servers.simulation.command': sys.executable,
              'mcp_servers.simulation.args': [str(Path(__file__).with_name('decision_mcp.py')), endpoint, token],
              'mcp_servers.simulation.required': True,
              'mcp_servers.simulation.tool_timeout_sec': 360,
              'mcp_servers.simulation.enabled_tools': ['submit_decision'],
              'mcp_servers.simulation.default_tools_approval_mode': 'approve'}
    for key, value in config.items():
        cmd += ['-c', key+'='+json.dumps(value, ensure_ascii=False)]
    for feature in ('memories', 'chronicle', 'apps', 'plugins', 'hooks', 'browser_use',
                    'computer_use', 'multi_agent', 'shell_tool', 'unified_exec', 'view_image'):
        cmd += ['--disable', feature]
    return cmd+['-']
