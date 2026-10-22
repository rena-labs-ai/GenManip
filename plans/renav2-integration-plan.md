# ReNav2 → GenManip Integration Plan

**Goal:** Integrate a custom mobile robot (`renav2`: 4WD base + 6-DOF arm + gripper) with 3 Arducam cameras (left base, right base, wrist) into GenManip, and generate VLA training data in LeRobot v2.1 format that matches the existing real-robot `renav2_pnp` dataset schema so sim and real data can be mixed.

**Status:** Ready to execute once the renav2 URDF is available. Each phase is self-contained and can be executed in a fresh session. Execute phases in order; each has a verification checklist that must pass before moving on.

---

## Target output schema (from real dataset `~/.cache/huggingface/lerobot/renav2/renav2_pnp_1/meta/info.json`)

| Feature | dtype | shape | names |
|---|---|---|---|
| `observation.state` | float32 | (7,) | joint_1..joint_6, gripper |
| `action` | float32 | (7,) | joint_1..joint_6, gripper |
| `observation.images.image_0` | video | (480, 640, 3) | left base cam, landscape |
| `observation.images.image_1` | video | (480, 640, 3) | right base cam, landscape |
| `observation.images.image_2` | video | (640, 480, 3) | wrist cam, **portrait** |
| `observation.task_done` | float32 | (1,) | per-frame done flag |

Plus: `fps: 25`, `robot_type: "renav2"`, codebase_version v2.1, av1-encoded video. **Base is NOT in the action space** — episodes are manipulation-only with a static base. This drives the key architecture decision: v1 uses a fixed base.

---

## Phase 0 — Consolidated findings: Allowed APIs & key facts

These were verified by reading source on 2026-06-11. Re-verify line numbers if the repo has moved (`git log --oneline -3`; plan written at commit `221553e`).

### Robot embodiment system
- `ManipRobotConfig` (`genmanip/core/robot/base.py:27-50`) — 9 **required** fields: `embodiment_name`, `arm_name`, `gripper_name`, `arm_dof_num`, `gripper_dof_num`, `gripper_open: List[float]`, `gripper_close: List[float]`, `default_arm_dof_indices: List[int]`, `default_gripper_dof_indices: List[int]`.
- `BaseEmbodiment.__init__` (`base.py:55-78`) calls `self.create_robot(scene_uid=kwargs["scene_uid"], default_config=kwargs["default_config"], robot_config=kwargs["robot_config"])`, then `self.set_planner(kwargs["default_config"]["current_dir"])` which must return a list of 1 or 2 `CuroboPlanner`s.
- Abstract methods to implement: `create_robot(scene_uid, default_config, robot_config) -> Robot` and `set_planner(current_dir) -> List[CuroboPlanner]`. `_plan_pose()` is already implemented by `SingleArmEmbodiment` (`genmanip/core/robot/singlearm_manip.py:14-32`) — do not reimplement.
- **Auto-discovery**: `genmanip/extensions/robots/__init__.py:1-11` walks the directory recursively and imports every `.py` file — a new file under `genmanip/extensions/robots/default/renav2/` registers automatically via its `@RobotFactory.register(...)` decorator. No central import list to edit.
- Instantiation: `genmanip/core/scene/scene.py:165-173` calls `RobotFactory.build(robot_config.type, scene_uid=..., default_config=..., robot_config=...)` per entry in task-config `robots:`.
- `RobotConfig` (`genmanip/core/scene/scene_config.py:11-21`): `type` (required), `default_joint_positions`, `position`, `orientation` (all optional; orientation is wxyz).
- Templates: single-arm fixed = `genmanip/extensions/robots/default/franka/robotiq.py` (reads scene placeholder `/World/{scene_uid}/franka` pose); mobile = `genmanip/extensions/robots/default/r5a/lift2.py:51-88` (uses `robot_config.position/orientation` with hardcoded defaults — **copy this placement pattern**).
- Base motion (only if needed later): `DualArmEmbodiment.delta_move_to()` (`genmanip/core/robot/dualarm_manip.py:113-125`) drives 3 *dummy virtual joints* (prismatic x, prismatic y, revolute yaw) via `robot_view.set_joint_position_targets(..., joint_indices=self.base_dof_indices)`. See `aloha_split/piper.py:43-45` for the dummy joint prim paths pattern.

### cuRobo planner
- `CuroboPlanner(robot_cfg: dict, robot_prim_path: str)` (`genmanip/utils/planner/curobo/base.py:30-86`). After construction set: `planner.ordered_js_names = [<6 arm joint names>]`, `planner.dof_len = 6`. `raw_js_names` only needed if USD joint names differ from URDF names (see `aloha_split/piper.py:123-132`).
- Config YAML schema: copy `configs/robots/curobo/franka_robotiq.yml` — it has **inline collision spheres** (lines 50-154), which is the pattern to follow (the external `spheres/*.yml` file referenced by `franka.yml` is not in the repo). Required keys under `robot_cfg.kinematics`: `urdf_path`, `usd_path`, `usd_robot_root`, `base_link`, `ee_link`, `collision_link_names`, `collision_spheres`, `collision_sphere_buffer`, `extra_collision_spheres: {attached_object: N}`, `self_collision_ignore`, `self_collision_buffer`, `mesh_link_names`, `lock_joints`, and `cspace:` (`joint_names`, `retract_config`, `null_space_weight`, `cspace_distance_weight`, `max_acceleration`, `max_jerk`).
- Path substitution pattern: aloha's `set_planner` (`piper.py:91-103`) replaces `${ASSETS_DIR}` in `usd_path`/`urdf_path` with `current_dir + "/saved/assets"`. Reuse this.

### Cameras
- Config YAMLs in `configs/cameras/`, referenced from task config via `domain_randomization.cameras.config_path`.
- **Intrinsics-based mode (use this — matches real Arducam calibration):** triggered by presence of `pixel_size` key. Keys: `pixel_size` (microns), `f_number`, `focus_distance` (m), `camera_params: [fx, fy, cx, cy]` (pixels), `resolution: [w, h]`, `position`, `orientation`, `camera_axes: usd`. Conversion math in `genmanip/utils/usd_utils/camera_utils.py:342-395`. Copy-ready example: `configs/cameras/fixed_camera_aloha_simbox_v2.yml` lines 28-89 (robot-mounted cameras with real intrinsics).
- Attachment: `exists: true` + `prim_path` resolves to `/World/{scene_uuid}{prim_path}` (`genmanip/utils/loader/scene.py:160-212`) — **a Camera prim must exist at that path inside the robot USD**; `position`/`orientation` are then applied via `set_local_pose` relative to the camera's parent prim.
- Annotator flags per camera: `with_distance`, `with_semantic`, `with_bbox2d`, `with_bbox3d`, `with_motion_vector` (set `false` for speed unless needed).
- Camera pose jitter: `domain_randomization.camera_randomization: {<cam_name>: {max_translation_noise: <m>, max_orientation_noise: <deg>}}` (`genmanip/utils/loader/domain_randomization.py:103-148`).

### Task config (`SceneConfig`, `genmanip/core/scene/scene_config.py:205-252`)
- Required: `task_name`, `usd_name`, `table_uid`, `robots`, `domain_randomization.cameras`. Useful: `num_episode`, `num_steps`, `instruction`, `object_config`, `layout_config`, `generation_config.goal`.
- `RandomEnvironmentConfig` (`scene_config.py:36-59`): `has_wall`, `hdr` (random HDR dome light from `{ASSETS_DIR}/miscs/hdrs/`), `robot_base_position` (bool or `{random_range: m}`), `robot_eepose`, `table_texture`, `table_type`, `wall_texture`, `room_randomization`, `random_visuals`.
- Templates: `configs/tasks/banana_robotiq_example.yml` (single-arm PnP), `configs/tasks/minimal.yml` (smallest), `configs/tasks/mobile_manip_demo.yml` (base_motion_list reference, only needed for optional Phase 10).

### Recording & conversion
- Recording is 30 Hz, **hardcoded** at `genmanip/demogen/workflow/demogen.py:210-211` (`physics_dt=1/30, rendering_dt=1/30`); `SceneConfig.physics_dt/rendering_dt` fields exist but are NOT passed there.
- LMDB keys (state via `genmanip/demogen/recoder/planning_recorder.py:236-340`): `arm_action`, `gripper_action` (per-DOF), `gripper_close` (scalar command), `base_motion`, `observation/robot/qpos`, `qvel`, `ee_pose_state`, `robot2env_pose`, per-object poses. Images via render_recorder: `observation/{camera_name}/color_image/{frame:04d}` as pickled JPG bytes.
- Converter `standalone_tools/genmanip2lerobot.py`: `ROBOT_CONFIGS` dict (franka_robotiq entry at lines 145-190 is the single-arm template), `fps=15` **hardcoded** at line 431, `TARGET_VIDEO_SIZE = (640, 480)` global at line 24, `_decode_color_frame` (lines 512-525) returns **BGR**, output feature names are GR00t-style (`state.joints`, `video.{cam}_view`). LeRobot pinned to commit `2b71789e15c35418b1ccecbceb81f4a598bfd883` (v2.1 API: `LeRobotDataset.create()`, `add_frame()`, `save_episode()`).
- **No per-frame `task_done` is recorded** — episode success is binary, evaluated by MetricsManager at episode end (`demogen.py` workflow line 323-324, `record_log(self.save_dir, "success")`).
- Inspection tools: `standalone_tools/data_vis.py` (LMDB → annotated mp4), `standalone_tools/verify/check_data_valid.py` (key/structure validation).

### Anti-patterns (do NOT do these)
- Do not pass `fps`, `physics_dt` overrides through the task YAML expecting them to take effect — the workflow hardcodes 1/30; patch code instead (Phase 7).
- Do not reference a camera `prim_path` that doesn't exist in the robot USD when `exists: true` — `Camera()` will fail or silently render nothing.
- Do not give the cuRobo config the chassis as `base_link` — it must be the arm's mount link; the chassis becomes collision geometry instead.
- Do not leave wheel joints articulated in the imported USD — they shift every DOF index and break `default_arm_dof_indices`.
- Do not invent `ManipRobotConfig` fields or omit any of the 9 required ones — Pydantic will raise at registry build time, inside Isaac Sim startup, which is slow to debug.
- Do not assume RGB from the converter — `cv2.imdecode` returns BGR; verify channel order against the real dataset in Phase 7.

---

## Phase 1 — Environment baseline (de-risk before any custom work)

**Goal:** Prove the stock Franka pipeline works end-to-end on this machine before touching anything renav2-specific.

### Tasks
1. Verify Isaac Sim python environment: `python -c "from isaacsim import SimulationApp; print('ok')"` (must run inside the Isaac Sim conda/venv this repo expects). Verify `python -c "import curobo, lmdb, lerobot"` .
2. Download baseline assets (HuggingFace-backed, `standalone_tools/download_assets.py`): run `download_basic_dataset()`, `download_banana_dataset()`, `download_robot_franka_robotiq_dataset()`. Assets land under `saved/assets/`.
3. Run baseline generation, headed, a few episodes:
   `python demogen.py -cfg configs/tasks/banana_robotiq_example.yml -l` (edit `num_episode` down to ~3 first).
4. Inspect output: find episodes under the demonstration dir (`{DEMONSTRATION_DIR}/{task_name}/trajectory/{timestamp}/lmdb`), run `standalone_tools/verify/check_data_valid.py --data_path <parent> --type trajectory` and `standalone_tools/data_vis.py --data_path <episode> --camera_name realsense --annotator_type st qp ins --output_path /tmp/baseline.mp4`.
5. Run stock conversion: `python standalone_tools/genmanip2lerobot.py --data_path <trajectory dir> --robot franka_robotiq --overwrite`; load the result with `LeRobotDataset` and print `info`.

### Verification checklist
- [ ] Isaac Sim launches headed and the Franka picks/places the banana.
- [ ] `check_data_valid.py` passes for all generated episodes.
- [ ] `data_vis.py` mp4 shows correct RGB rendering from all 3 stock cameras.
- [ ] Converted LeRobot dataset loads, has 7-dim `state.joints` and 3 `video.*_view` streams.

**If this phase fails, stop** — fix environment/assets first; nothing downstream can work.

---

## Phase 2 — Robot USD preparation (URDF → USD)

**Prerequisite:** renav2 URDF + meshes in hand. **Inputs to collect first:** exact arm joint names, gripper joint name(s) + open/close positions, wrist link name, chassis link name, real camera extrinsics (from URDF or CAD/calibration).

### Decisions baked in
- **v1 = fixed base.** The real dataset's action space has no base DOFs. A fixed-base articulation gives clean DOF indexing: arm joints 0-5, gripper 6+.
- Wheels: remove wheel joints (merge into chassis as fixed) in a *preprocessed copy* of the URDF before import, or convert them to fixed joints. Keep meshes for visuals.

### Tasks
1. Preprocess URDF copy: make all 4 wheel joints `type="fixed"`; confirm remaining movable joints = 6 arm + gripper joint(s) only.
2. Import via Isaac Sim URDF Importer (GUI: Isaac Utils → Workflows → URDF Importer, or `omni.importer.urdf` python API). Settings: Fix Base = **true**, merge fixed joints = **false** (keep link prims so cameras/cuRobo can reference them), default drive = position, units = meters.
3. Save as `saved/assets/robot_usds/renav2/robot.usd`. Note the root prim name (becomes `usd_robot_root` in the cuRobo config and the prefix for camera prim paths).
4. Add three **Camera prims** inside the USD (Create → Camera, then parent them):
   - `<root>/<chassis_link>/left_camera_mount/left_camera`
   - `<root>/<chassis_link>/right_camera_mount/right_camera`
   - `<root>/<wrist_link>/wrist_camera_mount/wrist_camera`
   Place mounts at the real extrinsic poses. (Reference: franka_robotiq USD has `/robotiq/arm/panda_link7/camera_mount/camera/realsense`.) Exact intrinsics/fine pose come from YAML in Phase 5 — prim existence and roughly-correct placement is what matters here.
5. In the GUI, verify joint drives: move each arm joint via the Articulation Inspector; tune drive stiffness/damping if the arm sags (pattern for per-joint drive fixes: `franka/robotiq.py:74-105` using `set_drive_damping_and_stiffness` / `set_drive_max_force` from `genmanip/utils/usd_utils`).
6. Record the **DOF order** of the articulation (Isaac Sim sorts DOFs; do not assume URDF order): open the robot in a scratch script, print `robot.dof_names`. Write the result into this plan file for Phase 4.

### Verification checklist
- [ ] USD opens in Isaac Sim; articulation has exactly `6 + gripper_dof` DOFs; `robot.dof_names` recorded.
- [ ] Base does not move when arm is commanded (fixed base confirmed).
- [ ] All 3 camera prims exist; through-camera viewport shows plausible left/right/wrist views.
- [ ] Gripper opens/closes by setting its joint position(s) to the URDF limits; note exact open/close values for Phase 4.

---

## Phase 3 — cuRobo configuration

**Goal:** `configs/robots/curobo/renav2.yml` that lets `CuroboPlanner` plan collision-free arm motion.

### Tasks
1. Copy `configs/robots/curobo/franka_robotiq.yml` → `configs/robots/curobo/renav2.yml`. Edit:
   - `urdf_path` / `usd_path`: point at the renav2 assets (use the `${ASSETS_DIR}` placeholder + the substitution pattern from `aloha_split/piper.py:97-102`, or absolute path under `saved/assets/`).
   - `usd_robot_root`: root prim from Phase 2.
   - `base_link`: the **arm mount link** (where the arm bolts to the chassis), NOT the chassis root.
   - `ee_link`: gripper TCP link (or last wrist link; adjust grasp offsets later).
   - `cspace.joint_names`: the 6 arm joint names; `lock_joints`: gripper joint(s) at open position.
   - `retract_config`: a sensible "home" pose (6 values); `null_space_weight`/`cspace_distance_weight`: `[1,1,1,1,1,1]` to start; `max_acceleration: 15.0`, `max_jerk: 500.0` (franka defaults).
2. Generate collision spheres. Use Isaac Sim's **Lula Robot Description Editor** (Isaac Utils menu) to fit spheres per link interactively, or cuRobo's `RobotConfig`/sphere-fit utilities. Paste results **inline** under `collision_spheres:` following `franka_robotiq.yml:50-154` format (`<link>: [{center: [x,y,z], radius: r}, ...]`).
3. Include the **chassis link** in `collision_link_names` with a few generous spheres so the planner never sweeps the arm through the base. Add `attached_object` to `extra_collision_spheres` (franka_robotiq uses 100).
4. Build `self_collision_ignore`: each link ignores its immediate neighbors (copy adjacency logic from franka_robotiq.yml).
5. Validate standalone (no full pipeline): scratch script that constructs `CuroboPlanner(load_yaml("configs/robots/curobo/renav2.yml"), <prim_path>)`, sets `ordered_js_names`/`dof_len=6`, then round-trips `fk_single`/`ik_single` and a `plan()` to a reachable pose.

### Verification checklist
- [ ] `CuroboPlanner` constructs without error (config schema valid).
- [ ] `fk_single(retract_config)` returns a pose matching the USD ee position (visually or numerically).
- [ ] `plan()` to a pose 30 cm in front of the robot returns a trajectory; to a pose inside the chassis returns `None` (collision spheres working).

### Anti-pattern guards
- No external `spheres/*.yml` reference (file pattern not shipped in repo) — inline only.
- Don't include gripper finger joints in `cspace.joint_names` — lock them.

---

## Phase 4 — Embodiment class

**Goal:** `genmanip/extensions/robots/default/renav2/renav2.py`, auto-registered as `manip/renav2/arm`.

### Tasks
1. Create the file by copying the structure of `franka/robotiq.py` but the **placement pattern of `r5a/lift2.py:51-88`** (use `robot_config.position`/`orientation` with defaults; deactivate the scene's `/World/{scene_uid}/franka` placeholder if present, as both templates do).

```python
@RobotFactory.register("manip/renav2/arm")
class Renav2Embodiment(SingleArmEmbodiment):
    def __init__(self, *args, **kwargs) -> None:
        config = ManipRobotConfig(
            embodiment_name="renav2",
            arm_name="<arm model name>",
            gripper_name="<gripper model name>",
            arm_dof_num=6,
            gripper_dof_num=<G>,                  # from Phase 2 dof_names
            gripper_open=[<...>],                  # G values, from Phase 2
            gripper_close=[<...>],                 # G values
            default_arm_dof_indices=[<6 ints>],    # positions of arm joints in robot.dof_names
            default_gripper_dof_indices=[<G ints>],
        )
        super().__init__(config, *args, **kwargs)
```

2. `create_robot()`: `create_prim` with `usd_path=os.path.join(default_config["ASSETS_DIR"], "robot_usds/renav2/robot.usd")` at `prim_path=f"/World/{scene_uid}/renav2"`, then `Robot(prim_path=..., name="renav2")`, solver iteration counts as in lift2 (`set_solver_position_iteration_count(128)`, `set_stabilization_threshold(0.005)`, `set_solver_velocity_iteration_count(4)`).
3. `set_planner()`: load `configs/robots/curobo/renav2.yml`, apply `${ASSETS_DIR}` substitution, create one `CuroboPlanner`, set `ordered_js_names` (6 arm joint names in cuRobo cspace order) and `dof_len = 6`. Set `raw_js_names` only if USD joint names differ from URDF names. Return `[planner]`.
4. If the gripper needs drive tuning (mimic joints, sagging fingers), add an `_initialize()` override following `franka/robotiq.py:74-108`.
5. **Important geometric check:** the arm mount on a 4WD chassis is low relative to GenManip's tabletop scenes. Set the default placement `position` z and x/y so the arm workspace covers the table surface used by the chosen scene (`robot_config.position` in the task YAML overrides this per-task). If the chassis is too short for any table scene, either choose lower scenes or place the robot on a riser — decide when first scene loads in Phase 6.

### Verification checklist
- [ ] `python -c "import genmanip.extensions.robots; from genmanip.core.robot.utils import RobotFactory; print(RobotFactory.get('manip/renav2/arm'))"` prints the class (auto-discovery worked). (Run inside the Isaac Sim env; imports pull omni modules.)
- [ ] In a headed scratch scene: robot spawns at requested pose, `embodiment.robot.dof_names` matches the indices in the config, gripper open/close via the embodiment's gripper values works.

---

## Phase 5 — Camera config from Arducam calibration

**Goal:** `configs/cameras/renav2_cameras.yml` with three cameras matching the real rig.

**Inputs to collect first:** Arducam intrinsics per camera (fx, fy, cx, cy at native capture resolution), real mounting extrinsics.

### Tasks
1. Copy `configs/cameras/fixed_camera_aloha_simbox_v2.yml` (lines 28-89 are the robot-mounted, intrinsics-based examples) → `configs/cameras/renav2_cameras.yml`.
2. Define three entries — `left_camera`, `right_camera`, `wrist_camera` — all with `exists: true` and `prim_path` matching the Camera prims created in Phase 2 (path **relative to** `/World/{scene_uuid}`, i.e. starts with `/renav2/...`).
3. Use the **intrinsics mode** per camera: `pixel_size` (Arducam sensor spec, e.g. OV9782 = 3.0 µm), `f_number`, `focus_distance`, `camera_params: [fx, fy, cx, cy]`, `resolution: [640, 480]` for base cams and `resolution: [480, 640]` for the wrist cam (portrait, matching `image_2`'s 640×480 portrait shape).
4. `position`/`orientation` per camera: fine-tuned extrinsics applied via `set_local_pose` relative to the mount prim. Orientation convention is ambiguous in the codebase (known gap) — resolve empirically: render, compare against a real frame from `renav2_pnp_1`, adjust.
5. Set annotators minimal for speed: `with_distance: false`, `with_semantic: false`, etc. (enable later only if you want depth/seg auxiliary supervision).

### Verification checklist
- [ ] All three cameras initialize in a generation run (no missing-prim errors).
- [ ] Side-by-side: a rendered frame per camera vs. a real frame from `renav2_pnp_1` videos — FOV and viewpoint geometry visibly match (objects at known distance subtend similar pixels).
- [ ] Wrist camera output is portrait 480 wide × 640 tall.

---

## Phase 6 — Task config + smoke-test generation

**Goal:** First renav2 episodes recorded to LMDB, randomization OFF.

### Tasks
1. Copy `configs/tasks/banana_robotiq_example.yml` → `configs/tasks/renav2_pnp_smoke.yml`. Edit:
   - `robots: [{type: manip/renav2/arm, position: [<x,y,z from Phase 4 check>], orientation: [<wxyz>]}]`
   - `domain_randomization.cameras.config_path: configs/cameras/renav2_cameras.yml`
   - all `random_environment` flags `false`, `camera_randomization: null`
   - `num_episode: 5`, an `instruction` matching your real-task phrasing (e.g. "pick up the X and place it on the Y")
   - keep the banana scene/objects (`usd_name`, `object_config`, `table_uid`, `layout_config`) for the smoke test — known-good assets.
2. Run headed: `python demogen.py -cfg configs/tasks/renav2_pnp_smoke.yml -l`. Watch the full pick-and-place; debug planner failures here (usually collision spheres or `ee_link` grasp offset — revisit Phase 3 values).
3. Run headless for the remaining episodes: `python demogen.py -cfg configs/tasks/renav2_pnp_smoke.yml`.
4. Validate: `check_data_valid.py`, then `data_vis.py --camera_name wrist_camera --annotator_type st qp aa ins` and inspect all three camera streams.

### Verification checklist
- [ ] ≥3/5 episodes succeed (MetricsManager logs "success").
- [ ] LMDB contains `observation/robot/qpos` with `6 + gripper_dof` columns, `arm_action` with 6, `gripper_action`, and `observation/{left,right,wrist}_camera/color_image/*` keys.
- [ ] data_vis videos show correct camera views and a smooth, collision-free arm trajectory.

---

## Phase 7 — LeRobot conversion matched to `renav2_pnp` schema

**Goal:** Converted sim episodes indistinguishable in schema from the real dataset.

**Decision:** Fork rather than patch — create `standalone_tools/genmanip2renav2lerobot.py` as a copy of `genmanip2lerobot.py`. Four things differ from the stock script (feature naming, fps, per-camera target size/orientation, task_done injection); a fork keeps the stock converter working for other robots and is easier to verify.

### Tasks
1. **First, inspect the real dataset's conventions** (ground truth for every choice below):
   - `task_done` semantics: load a few episodes of `renav2_pnp_1`, plot `observation.task_done` over time (always 0? steps to 1 at success?). Mirror exactly.
   - Gripper value range in `observation.state[6]` (radians? 0-1 normalized?) — define the sim→real gripper mapping accordingly.
   - Action semantics: real `action` is presumably commanded joint targets; sim equivalent is `arm_action` (+ gripper command from `gripper_close`). Confirm real action ≈ next-frame state (target positions) vs. something else.
2. In the fork, add the robot entry:
   ```python
   "renav2": {
       "robot_type": "renav2", "is_dual_arm": False,
       "joint_indices": [<6 arm cols in qpos>], "gripper_indices": [<gripper col(s)>],
       "base_indices": [],
       "cameras": ["left_camera", "right_camera", "wrist_camera"],
   }
   ```
3. Replace the features dict with the real schema: `observation.state` (7,), `action` (7,), `observation.images.image_0/1/2` with HWC shapes `(480,640,3)/(480,640,3)/(640,480,3)`, `observation.task_done` (1,). Names exactly `["joint_1",...,"joint_6","gripper"]`.
4. State/action assembly per frame: `state = concat(qpos[joint_indices], gripper_scalar(qpos[gripper_indices]))`; `action = concat(arm_action, gripper_scalar_command)` where `gripper_scalar(...)` implements the mapping from step 1 (for a multi-DOF gripper, collapse to one scalar, e.g. normalized first-finger position).
5. fps: real data is 25 fps; recording is 30 Hz.
   - Preferred: patch `genmanip/demogen/workflow/demogen.py:210-211` to `physics_dt=1/25, rendering_dt=1/25` (or pass `self.scene_config.physics_dt` through) and set `fps=25` in `LeRobotDataset.create`. Re-generate Phase 6 data after this change.
   - Fallback if 25 Hz physics destabilizes grasping: keep 30 Hz, resample 30→25 in the converter (or accept fps=30 and let training-time delta-timestamps handle it — but then sim/real mixing needs care).
6. Cameras: per-camera target size instead of global `TARGET_VIDEO_SIZE` — image_0/1 → (640,480); image_2 → (480,640). Convert BGR→RGB (`cv2.cvtColor`) — check against real video channel order before assuming.
7. `task_done`: populate per frame according to the convention from step 1, using episode success (only convert successful episodes; success is in the episode logs) and, if real data flips at completion time, derive the flip frame from the metrics or last-N-frames heuristic — match whatever real data does.
8. Convert the Phase 6 episodes: `python standalone_tools/genmanip2renav2lerobot.py --data_path <trajectory dir> --robot renav2 --overwrite`.

### Verification checklist
- [ ] `meta/info.json` of sim output: features/names/shapes/fps/robot_type **diff clean** against `renav2_pnp_1/meta/info.json` (write a 10-line diff script; only totals should differ).
- [ ] Load both datasets with the same `LeRobotDataset` code; state/action ranges (per-joint min/max) are comparable; gripper range matches.
- [ ] Decode frame 0 of each video stream from both datasets; colors are RGB-correct and orientations match (wrist portrait).
- [ ] A `lerobot` visualization or your training dataloader accepts the sim dataset without code changes.

---

## Phase 8 — Domain randomization & scale-up

**Goal:** The diversity that motivated using GenManip, then volume.

### Tasks
1. Copy the smoke config → `configs/tasks/renav2_pnp_dr.yml`; enable incrementally (one flag per short run, verify renders each time): `hdr: true` → `table_texture: true` → `table_type: true` → `wall_texture/has_wall` → `room_randomization: true`.
2. Add camera jitter matching real-world remount tolerance:
   ```yaml
   camera_randomization:
     left_camera:  {max_translation_noise: 0.01, max_orientation_noise: 2.0}
     right_camera: {max_translation_noise: 0.01, max_orientation_noise: 2.0}
     wrist_camera: {max_translation_noise: 0.005, max_orientation_noise: 2.0}
   ```
3. Object diversity: download the Objaverse subset (`download_objaverse_subset_dataset()` in `standalone_tools/download_assets.py`), then use `object_config` with `type: load_object_from_path` / `uid_list` pools and `layout_config` ranges (`random_obj1_range` template in `banana_robotiq_example.yml`; full schema `scene_config.py:167-202`). Reference `configs/tasks/scaling_up/*.yml` for large object-pool configs.
4. Robot pose diversity: `robot_base_position: {random_range: 0.05}` — randomizes the (static) base placement per episode, which is exactly the variation a parked 4WD base would have in reality.
5. Multiple instructions/tasks: one task YAML per task template; vary `instruction` and goal objects to mirror your real `total_tasks` structure.
6. Scale: headless runs, `num_episode` to target count; only successful episodes convert. For multi-GPU/parallel later: `ray_eval_server.py` exists (out of scope here).
7. Convert everything with the Phase 7 script; spot-check 5 random episodes with the Phase 7 checklist.

### Verification checklist
- [ ] Renders show varied lighting/textures/rooms across episodes (eyeball 10 episodes' first frames).
- [ ] Success rate under full randomization is acceptable (>50%; if it collapses, narrow layout ranges).
- [ ] Converted dataset passes the Phase 7 schema diff.

---

## Phase 9 — Final verification

1. Re-run all phase checklists' key items on the final dataset (schema diff, ranges, channel order, fps).
2. Anti-pattern grep: confirm no stock-file regressions — `git diff --stat` should touch only: new files (`extensions/robots/default/renav2/`, `configs/robots/curobo/renav2.yml`, `configs/cameras/renav2_cameras.yml`, `configs/tasks/renav2_*.yml`, `standalone_tools/genmanip2renav2lerobot.py`) plus the single dt patch in `demogen/workflow/demogen.py`.
3. Mixed-loader test: instantiate your VLA training dataloader over `renav2_pnp_1` + the sim dataset together; iterate 100 batches without error; visually inspect a few mixed batches.
4. Sanity-train: a short fine-tune on sim-only data; confirm loss decreases and rollouts in sim look sane (optional but recommended before generating tens of thousands of episodes).

---

## Phase 10 (optional, later) — Mobile base motion

Only if you decide to put base motion in the action space (your real dataset doesn't have it today).

- Re-import URDF with 3 dummy virtual joints (prismatic x, y + revolute yaw between world and chassis), following the aloha USD pattern (`dummy_base_x/mobile_translate_x` etc., `aloha_split/piper.py:43-45`); arm/gripper DOF indices shift by +3 — update `ManipRobotConfig` and converter indices.
- Port `delta_move_to` from `DualArmEmbodiment` (`dualarm_manip.py:113-125`) into the renav2 embodiment (single-arm + base hybrid; the `move_to` skill currently type-checks for dual-arm, so either subclass or adapt the skill).
- Recording already supports `base_motion` (3-dim deltas) and the stock converter shows the `state.base`/`action.base` pattern (see `aloha_split` entry in `ROBOT_CONFIGS`).
- Note: this is kinematic base teleportation via joint targets, not 4WD dynamics — fine for VLA data, not for navigation realism.

---

## Open questions to resolve when the URDF arrives

1. Gripper: how many actuated DOFs, mimic joints?, open/close joint values, and the scalar convention used in the real dataset's `state[6]`.
2. Arm joint names and whether USD import renames them (affects `ordered_js_names` vs `raw_js_names`).
3. Arm mount height vs. GenManip table scenes — riser or scene choice (Phase 4, task 5).
4. Real `action` semantics (commanded targets vs. measured next-state) — Phase 7, task 1.
5. Arducam model/sensor → `pixel_size` value (e.g. OV9782 global shutter = 3.0 µm).
6. `task_done` convention in the real data — Phase 7, task 1.

## Estimated effort

| Phase | Effort | Risk |
|---|---|---|
| 1 Baseline | 0.5 day | env/asset download issues |
| 2 USD | 0.5-1 day | importer quirks, drive tuning |
| 3 cuRobo | 1-2 days | **highest risk**: collision spheres + planner tuning |
| 4 Embodiment | 0.5 day | low (template copy) |
| 5 Cameras | 0.5 day | orientation convention trial-and-error |
| 6 Smoke gen | 0.5-1 day | grasp offsets, reachability |
| 7 Conversion | 1 day | schema fidelity details |
| 8 DR + scale | 1 day + compute | success-rate tuning |
| 9 Verify | 0.5 day | — |
