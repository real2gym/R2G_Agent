# Environment and scene setup

Run from the repository root after installation. Supply your own absolute paths and live service URLs; placeholders below are descriptive, not deployed endpoints.

```bash
export R2G_TASK_ID=E03
export R2G_TASK_LANGUAGE='Your task instruction and predeclared completion requirements'
export R2G_SCENE_DIR=/path/to/matching/scene
export R2G_CRITERIA_PATH=/path/to/frozen/criteria.json
export R2G_RUN_ROOT=/path/to/new/episode
export R2G_MODEL_CLI=/path/to/compatible/model-cli
export R2G_MODEL=your-accessible-model
export R2G_SIM_PYTHON="$(command -v python)"
export R2G_BACKEND_MODULE=roboagent.backends.ego_backend
export R2G_BACKEND_CONFIG="$PWD/examples/backend.json"
export R2G_POLICY_PROMPT="$PWD/roboagent/resources/ego_policy.txt"
export R2G_SAM3_URL=http://your-sam3-service:8114
export R2G_GRASPNET_URL=http://your-graspnet-service:8115
export R2G_CRITERIA_SHA256="$(python -c 'import hashlib,os,pathlib; print(hashlib.sha256(pathlib.Path(os.environ["R2G_CRITERIA_PATH"]).read_bytes()).hexdigest())')"
```

For headless Linux, set `MUJOCO_GL=egl` when your drivers support it. Do not set it blindly on other platforms.

Check the model CLI's `exec --help` and configuration support against `roboagent/command.py`. This code retains the experiment's CLI contract rather than claiming compatibility with every public binary. Model authentication is configured outside this repository. Services must expose the routes in `perception.md`; compatible port numbers alone are insufficient.

Verify the scene contains its XML (`model.xml` for D02; `scene.xml` otherwise), referenced assets, and an initial-state file whose qpos/qvel lengths match the model. E-series scenes expect FR3 joint, hand actuator and TCP names; D-series scenes expect the experiment's Franka/Robotiq names. See each task profile for the full naming contract. A different model requires an explicit adapter and independent validation, not renamed task labels.

Criteria JSON is read by `ego_profile.evaluate` or `batch_profile.evaluate`. It supplies `task_id`, `thresholds`, `required`, `settle_s`, `stable_s` and task-specific fields. The `required` set must match the implemented checks. The controller instruction and criteria must describe the same task. No example threshold set is invented here, since thresholds depend on the supplied reconstructed geometry.

The example runs the backend locally through `python -m`. Remote deployment is possible through an explicit operator-owned argv and SCP artifact configuration, but this release does not include a host, SSH account or remote bootstrap. The backend interpreter must have this package and simulation dependencies installed, and receive the required environment variables. The local template needs no SSH.

Run configuration validation, then launch only when all dependencies are ready. Both controller and backend outputs should be new; never point `artifact_dir` at source assets. Changing the model, scene, criterion or dependency environment changes the experiment and requires fresh validation.

For D01–D08 use `R2G_BACKEND_MODULE=roboagent.backends.droid` and the task-specific prompt listed in `scene-adapters.md`. D01 instead uses `examples/backend-d01.json` (no external criterion fields); D02–D08 use the regular backend template. Their initialization/evaluation contracts differ from the batch profile, including D02 wrist mounting, D05 elastic-band samples, and D06–D08 articulated contact checks.
