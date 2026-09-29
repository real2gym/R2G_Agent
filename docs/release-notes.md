# Source and release scope

Primary source: `C0_serial_E03_E12_20260925/source/C0/roboagent`.
Secondary comparison: `C0_serial_D09_E02_20260925/source/C0/roboagent`.
All shared modules were identical; the primary snapshot additionally supplied the dual-arm backend and profile. Robot-specific policy API text was taken from those snapshots' E03, D09, E01 and E02 policy files, rather than the older LIBERO prompt with different TCP conventions.

The earlier `R2G_Agent_C0` publication draft was used only for packaging/contributor conventions and adapted protocol tests. Its tests had stale per-round mode assumptions; the release tests use the actual continuous-stage protocol. See `source_manifest.json` for the original file identities and digests.

## Changes

- Exported only the continuous stage runner, decision transport, prompt/schema, input/code guards, task profiles and real MuJoCo backends.
- Removed prior-procedure loading, prompt injection, decision metadata, extraction/update/accumulation paths and their unused branches/dependencies.
- Extracted the CLI command constructor and single-tool stdio bridge so the agent no longer imports unrelated autonomous-tool orchestration.
- Moved actual SAM3/GraspNet HTTP adapters into a shared module, made endpoints explicit and required, and retained service failures rather than inventing observations.
- Added environment expansion with failure on unset variables; strict episode top-level keys; installed-module backend imports; public package metadata; exact decision schemas and regression tests.
- Removed unreachable fallback evaluation from the batch/ego backends and legacy defaults; retained the actually used D01 containment evaluator in the dedicated D01–D08 backend. Retained native task evaluators and robot-specific control behavior, including task-specific actuator/index assumptions.

## Limits

No weights, scene assets, generated results or private runtime patches are distributed. The live model CLI, perception servers and matching scenes were not exercised for this publication task. Unit/contract tests are not simulator success evidence. The API bounding-box helper intentionally reports axis-aligned conservative extents, as in the source, despite its historical function name.

The model context and code namespace persist only within the current episode. The runner withholds independent evaluation from feedback. The policy is still generated Python executed inside the simulator process; its AST guard is not operating-system isolation. The original native evaluator semantics were not redesigned during this export.

The overview SVG is maintained separately by the project author. Its paper-level labels do not introduce additional executable functionality into this repository. No commit or push was performed as part of this export.

## Verification on export

- `python3 -m compileall -q roboagent`: passed.
- `python3 -m unittest discover -s tests -v`: 17 tests passed (Python 3.9.6).
- Built `real2gym_agent-0.1.0` wheel and inspected its 45 entries, including resources and all three native adapters. Installed into an isolated local target and verified CLI help and configuration-only preflight.
- Original 36 copied-source hashes unchanged. Code/configuration/prompt/schema scan confirms complete removal of the retired prior-procedure interface; public text scan found no personal filesystem paths, private host aliases or embedded credentials.
- Temporary wheel/install/cache files removed. The author-managed SVG was not modified by this code export.
- Live model requests, SAM3/GraspNet GPU services and native task execution were not run.

## D01–D08 coverage correction

Audited actual per-scene C0 launch configs and verified all seven native source modules against the launch location. Added the independent D01–D08 backend, all six initialization/evaluation modules and eight robot-specific prompts. Shared real perception clients and environment expansion match the other adapters. D02 wrist mounting, D05 elastic-band logic, D06–D08 articulated/contact checks and D01 inline evaluation remain intact. See `scene-adapters.md`. No result files were imported. New tests cover retained stage/perception integration and criterion tamper rejection. Rebuilt wheel includes all additions. Live scene execution remains untested in this publication task.
