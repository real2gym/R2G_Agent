# Scene adapter coverage: all 24 C0 tasks

D01–D08 were audited against each task's actual C0 episode/backend configuration, not inferred from filenames. Every launch uses `publish_frameworks_20260924/C0/roboagent/backends/droid.py`. That file and its six supporting modules match the September 25 C0 snapshot byte-for-byte before release portability edits. No experiment results, trajectories, assets or private launch paths were copied.

All adapters share the continuous stage runner, decision format, persistent current-episode code namespace, 25/4000/600 budgets, real SAM3/GraspNet HTTP contracts, native motion and feedback transport. Differences are native scene contracts and evaluators; silently substituting `batch_backend` would lose behavior.

| Task | Initialization and native scene differences | Evaluation retained | Prompt |
| --- | --- | --- | --- |
| D01 | `scene.xml`, `initial_qpos.npy`, `scenario.json`; `fr3_base`/`rq_pinch` | Original inline block-in-cup containment, release and settling test; no external criterion JSON | `droid_policy.txt` |
| D02 | `model.xml`, `initial_state.json`, `scenario.json`; marker/driver state and `link0`/`pinch`; frame-zero wrist extrinsic converted to rigid `link7` mounting | Marker grasp → lift → carry → release, supported stable relocated pose | `droid_d02_policy.txt` |
| D03 | `scene.xml`, `initial_state.npz`, `scenario.json`; `milk_carton`, `milk_free`, `fr3_link0`/`rq_pinch` | Same ordered transport evaluator with carton footprint and configured upright orientation | `droid_d03_policy.txt` |
| D04 | `scene.xml`, `initial_state.npz`; `target_green`, `target_free`, `support_table_display` | Configured relocation, support and target geometry/orientation | `droid_d04_policy.txt` |
| D05 | `scene.xml`, `initial_state.npz`; 64 elastic-band nodes; specialized sampling interval | Native transport, winding/encirclement, bottle contact, release and stable band | `droid_d05_policy.txt` |
| D06 | `scene.xml`, `initial_state.npz`; `pd_link0`, `drawer_slide`, `gray_lower_drawer`; closed position from joint range | Contact-supported drawer closing, release/retraction and stability | `droid_d06_policy.txt` |
| D07 | `scene.xml`, `initial_state.npz`; `link0`, `cabinet_door_hinge`, `active_door`; retained scene-specific closed-angle offset | Partial rotation with required remaining opening, native push contact, release and stability | `droid_d07_policy.txt` |
| D08 | `scene.xml`, `initial_state.npz`; `link0`, `door_hinge`, `active_door`; retained scene-specific closed-angle offset | Full closing with angle tolerance, native push contact, release and stability | `droid_d08_policy.txt` |
| D09–D12, E01–E02 | `batch_backend`: task-specific actuator indices and object sets | Batch task trace evaluators | `batch_policy.txt`, `e01_policy.txt`, `e02_policy.txt` |
| E03–E12 | `ego_backend`: explicit dual-arm selection, per-arm bases/TCPs and held inactive arm | Bimanual task trace evaluators | `ego_policy.txt` |

D01 does not provide the contact-sensor helper used by the profiled tasks; calling it raises an explicit error. Its original final-state containment criterion is narrower than the later ordered-contact trace checks. This distinction is preserved rather than presenting every task as using identical evidence.

The first seven qpos/actuators and gripper slot 7 are part of the D01–D08 robot contract. Their profile code also contains retained model-specific constants (band node count, articulated closed-angle offsets and geometry names). They are not generic URDF adapters. To use different geometry, explicitly adapt and validate those contracts; do not replace them with guessed defaults.

D02–D08 criteria use their original `thresholds` and `required` schemas, which differ from `batch_profile`/`ego_profile`. Scene assets and criteria remain external. SHA-256 checks and evaluator-process separation are retained. Profiles are outer-framework code and are never supplied to the policy.

Publication validation covers source identity, module/prompt coverage, stage/perception integration, criterion tamper rejection and packaging. It does not rerun all 24 physical tasks or assert native success.
