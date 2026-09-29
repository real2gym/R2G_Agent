"""Unified public native backend: python -m roboagent.backends CONFIG."""
import argparse
import runpy
from .routing import adapter_for, load_backend_config


def main(argv=None):
    parser = argparse.ArgumentParser(description='Real2Gym native backend, routed by explicit scene_id')
    parser.add_argument('config', help='JSON backend configuration')
    args = parser.parse_args(argv)
    config = load_backend_config(args.config)
    # Preserve the native stdin/stdout JSON protocol. The validated configuration
    # is passed in memory, without re-reading the file or starting an extra worker.
    runpy.run_module(adapter_for(config['scene_id']),
                     init_globals={'BACKEND_CONFIG': config},
                     run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
