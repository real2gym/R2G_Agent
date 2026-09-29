# Real2Gym Agent

[Project](https://real2gym.github.io/) · [Real2Sim](https://github.com/real2gym/Real2Gym) · [Agent](https://github.com/real2gym/R2G_Agent) · [Paper](https://github.com/real2gym/Real2Gym/blob/main/paper/Real2Gym.pdf)

A simulation agent that turns current visual observations into executable manipulation stages, checks the actual feedback, and continues in one task-scoped model session.

![Agent overview](assets/agent-overview.png)

[Vector version (SVG)](assets/agent-overview.svg)

This release contains the **C0 continuous stage-code execution path** used in the September 25 experiments. It provides observation, planning, SAM3/GraspNet service adapters, native MuJoCo control, feedback and independent evaluation. It does not load previous-task procedures or build a persistent experience library. The overview illustrates the broader paper workflow; the executable scope here is the stage loop.

## Highlights

- **Closed-loop stages:** one persistent model context per task; bounded multi-action Python per decision.
- **Current evidence:** external and wrist RGB, calibrated metric RGB-D, robot state and explicit gripper contact feedback.
- **Real execution:** robot-only IK followed by native actuator commands; no target-object teleportation or motion replay.
- **Separate evaluation:** task-specific native trace checks remain in the outer framework; no hidden success or target poses enter model feedback.
- **Traceable source:** a minimal export of the actual experiment code, with original file hashes in [source provenance](docs/source_manifest.json).

## Execution loop

| Step | What happens |
| --- | --- |
| Observe | Acquire current external/wrist RGB and robot self-state. Stage code can request calibrated RGB-D. |
| Decide | Submit a stage plan, executable Python, expected visual effect and uncertainty through `submit_decision`. |
| Execute | Validate generated code; invoke SAM3/GraspNet, geometric helpers and bounded native motion. |
| Check | Return actual stdout/errors, self-state and fresh RGB to the same model session. |
| Finish | Stop or exhaust the budget, then evaluate the native trace outside the policy. |

The experimental limits are **25 decisions, 4,000 control steps per episode, and 600 control steps per stage**. A control step targets 20 ms; physics substeps depend on the model timestep. Commands do not prove successful contact or task completion.

## Install

Python 3.9 or newer is required. Use a dedicated simulation environment:

```bash
git clone https://github.com/real2gym/R2G_Agent.git
cd R2G_Agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[sim,test]'
python -m roboagent --help
```

The outer controller itself uses the Python standard library. The `sim` extra installs NumPy, SciPy, MuJoCo, Pillow and ImageIO/FFmpeg support. It does **not** install model services or their weights.

## Required external components

| Component | Requirement |
| --- | --- |
| Model CLI | An authenticated CLI compatible with the retained experimental command and JSON/MCP event protocol. The original run selected `gpt-6-astra`, medium effort; choose a model available to your account. |
| CLI compatibility | The launcher retains `--ignore-user-config`, `--ignore-rules`, `--ephemeral`, read-only sandbox and feature-disable flags. Some distributions do not support them. This repository does not bundle the experimental CLI patches; inspect `roboagent/command.py` and verify support before launching. No automatic weaker fallback is used. |
| Native scenes | A matching scene XML (`model.xml` for D02, otherwise `scene.xml`), referenced meshes/textures and task-specific initial-state files, including the exact robot/body/site naming expected by the adapter. |
| Criteria | For D02–D12/E01–E12, a task-matched, externally prepared JSON criterion file and its SHA-256. D01 retains its original inline containment criterion. Evaluator thresholds must be fixed before execution. |
| Perception | Running SAM3 and GraspNet-compatible services with the [documented HTTP contracts](docs/perception.md). Checkpoint access, GPU runtime and any access credentials are operator-provided. |
| Rendering | Working MuJoCo rendering, platform graphics drivers and a working FFmpeg encoder. Headless Linux commonly uses `MUJOCO_GL=egl`. |

Neither scene assets, checkpoint files, experimental outputs nor account configuration are included. **This is not a turnkey reproduction without those components.** Dependency ranges describe required APIs, not a recovered, fully pinned experiment environment.

## Configure and run

Start with [the environment/configuration guide](docs/setup.md). The templates use explicit `${R2G_...}` variables; unset variables fail validation. They contain no embedded machine paths or service hosts.

```bash
python -m roboagent preflight examples/episode.json
python -m roboagent run examples/episode.json
```

`preflight` validates and prints the expanded configuration only. It does not establish GPU, model authentication, model-service readiness or native task success. `run` requires a **fresh output directory**.

All scenes use one public native-backend entry point:

```bash
python -m roboagent.backends examples/backend.json
```

Normally the controller starts this command through `examples/episode.json`; do not start a second backend for the same output directory. The standalone command speaks the controller's newline-delimited JSON protocol on stdin/stdout; it is not an interactive policy run.

Set **`scene_id`** in the backend configuration (the templates use `R2G_SCENE_ID`). The dispatcher chooses the adapter automatically; no backend-module environment variable is needed. Shared fields are `scene_id`, `scene_dir`, `artifact_dir`, `language` and `max_steps` (4,000). All scenes except D01 also require `success_criteria_path` and `success_criteria_sha256`. D01 uses `examples/backend-d01.json` because its original evaluator is inline.

Select only the robot-appropriate policy prompt:

| Scene | Prompt under `roboagent/resources/` |
| --- | --- |
| D01 | `droid_policy.txt` |
| D02–D08 | `droid_d02_policy.txt` … `droid_d08_policy.txt` |
| D09–D12 | `batch_policy.txt` |
| E01 / E02 | `e01_policy.txt` / `e02_policy.txt` |
| E03–E12 | `ego_policy.txt` |

The internal adapters remain different because robot joint/TCP names, actuator layouts, wrist mounting, dual-arm control and native evaluation differ across the reconstructed scenes. These are implementation details behind the unified command, not different agent algorithms. All 24 scenes share the C0 stage loop and perception clients. [Per-scene contracts and routing](docs/scene-adapters.md) explain the retained differences.

The controller/criterion wire format retains `task_id` and `suite="droid"` for compatibility; the episode template fills `task.task_id` from the same `R2G_SCENE_ID`. Backend input requires `scene_id`; a matching legacy `task_id` is allowed, but conflicting IDs or a missing `scene_id` fail before simulator startup. Relative backend paths resolve against the JSON file's directory.

## Outputs and boundaries

A run writes controller decisions, prompts, hashes and backend feedback; fresh observation PNGs; two simulator videos; and an independently computed `native_result.json`. The bundled HTML viewer reads `public/run.json`; serve the generated public directory over local HTTP to view it. The backend also retains the native trace and final state. None of these generated files belong in this source repository.

The code preserves the experimental task evaluators. They inspect privileged simulator state **after execution** and must not be used as policy input. Explicit gripper pad-force feedback is available, so this is an RGB-D and contact-sensor-assisted protocol, not an RGB-only policy.

AST validation and an allowlisted Python namespace are **not a hardened sandbox**. The simulator executes model-generated code in-process. Use a separately managed, isolated research runtime without unrelated files or credentials. CLI isolation flags and an empty working directory do not replace operating-system containment.

## Development

```bash
python -m compileall -q roboagent
python -m unittest discover -s tests -v
```

Tests cover decision/schema removal, a single-session stage loop, withheld native feedback, serialization, portable configuration, and perception-service request/failure contracts. Tests use isolated protocol fixtures; they do not establish physical task success or live model/service compatibility. See [release notes](docs/release-notes.md) for the exact scope and remaining dependencies.
