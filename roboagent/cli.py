"""Public entry point for the continuous C0 stage loop."""
import argparse
import json
from pathlib import Path
from .config import load_config, validate_episode
from .runner import run

def main():
    parser = argparse.ArgumentParser(description='Real2Gym simulation stage-code agent')
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('preflight', 'run'):
        sub.add_parser(name).add_argument('config')
    args = parser.parse_args()
    path = Path(args.config).resolve()
    config = validate_episode(load_config(path))
    if args.action == 'preflight':
        print(json.dumps({'configuration': config,
            'note': 'Configuration validation only; no model, perception, rendering or simulator service was contacted.'}, indent=2))
        return
    print(run(config, path.parent))
