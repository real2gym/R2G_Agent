# Perception service contract

The client is extracted from the actual MuJoCo executors. It does not ship SAM3, GraspNet, checkpoint weights or server implementations. Provision compatible services independently. URLs come from `R2G_SAM3_URL` and `R2G_GRASPNET_URL`; both are required. HTTP failures propagate to stage feedback. No mock or fixed-pose fallback is used in production.

All requests are JSON POSTs with a 30-second client timeout. Binary arrays use base64; NumPy arrays use `.npy` with pickle disabled.

| Service route | Request | Response |
| --- | --- | --- |
| SAM3 `/segment` | `image_base64`: PNG; `text_prompt`: string | `results`: list with `mask_base64`, `shape`, optional `score` and `box` |
| SAM3 `/segment_point` | PNG `image_base64`; `point_coords`: `[u,v]` | `masks_base64`: raw array bytes; `masks_dtype`, `masks_shape`, `scores` |
| GraspNet `/plan` | `.npy` base64 `depth_base64`, `cam_K_base64`, `segmap_base64`; integer `segmap_id` | `.npy` base64 `grasps_base64` and `scores_base64` |

Text masks may be raw uint8, `.npy` or PNG with the declared shape. The Python API returns boolean masks. Optional grasp parameters retained from the experiment: `local_regions`, `filter_grasps`, `skip_border_objects`, `z_range`, `forward_passes`, `max_retries`.

Depth is metric optical z. Intrinsics are 3×3. Returned grasp transforms are N×4×4 in the optical camera frame; convert using observation `pose_mat @ grasp`. TCP quaternions in the control API are **wxyz**, while SciPy uses xyzw. E-series arm selection changes the robot-base frame; reacquire observations after switching arms.

This HTTP contract is deployment-specific. A third-party package called SAM3 or GraspNet is not automatically wire-compatible. Service model versions, weights, licensing/access and GPU requirements are external deployment responsibilities. Do not put credentials into committed endpoint URLs.
