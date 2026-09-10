# GenManip → Isaac Sim 6.0.1 Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run GenManip's stock Franka banana pick-and-place end to end (demogen → render → validate) on Isaac Sim 6.0.1 with PhysX, on an RTX 5090 laptop.

**Architecture:** Mechanical rename of `omni.isaac.*` imports to the Isaac Sim 4.5+ `isaacsim.*` names (still shipped, deprecated, in 6.0), a new Python 3.12 / torch cu128 environment, cuRobo rebuilt for sm_120, then runtime fixes discovered by running the baseline. No architectural changes to GenManip.

**Tech Stack:** Python 3.12, uv, torch 2.11.0+cu128, `isaacsim[all,extscache]==6.0.1.0`, cuRobo (source), PhysX, RTX Real-Time 2.0.

**Spec:** `plans/2026-09-10-isaac-6-port-design.md`

## Global Constraints

- Python 3.12 only; venv at `GenManip/.venv`; never touch system CUDA or use sudo.
- Physics backend stays PhysX; do not enable `isaacsim.physics.newton` or `isaac-sim.newton.sh`.
- Only rename imports in upstream files; put any new helper in a new file so `git merge upstream/main` stays cheap.
- Branch `port/isaac-6.0`; push to `origin` only, never `upstream`.
- Every task ends with a runnable check and a commit. No Claude attribution trailers (user's global rule).
- Headless by default (`demogen.py` without `-l`). `OMNI_KIT_ACCEPT_EULA=YES` in the environment.

## How this plan is meant to be run (the agentic project pattern)

1. **One task = one fresh subagent.** Give it only: this file's task text, the spec path, and the exact verification command. Fresh context per task avoids stale assumptions.
2. **Verification is the gate, not the report.** A task is done when its check command's output is pasted back and matches "Expected". Never accept "should work".
3. **Long-running steps go to background agents** (env install, asset download, first render) and the orchestrator continues with independent tasks.
4. **Every workflow that took more than one attempt becomes a skill** (Task 8) so the next robot, machine, or person does not re-derive it.
5. **Facts that are not derivable from the repo go to memory / CLAUDE.md** (hardware pins, version blockers), never into chat only.

---

### Task 0: Branch hygiene

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1:** `git checkout -b port/isaac-6.0` (done), then append `.venv/` to `.gitignore`.
- [ ] **Step 2:** Run `git status --short`; Expected: only `.gitignore` modified and `plans/` untracked.
- [ ] **Step 3:** `git add .gitignore plans/2026-09-10-isaac-6-port-design.md plans/2026-09-10-isaac-6-port-plan.md && git commit -m "port: Isaac Sim 6.0 design + plan, ignore .venv"`

---

### Task 1: Python 3.12 environment with Isaac Sim 6.0.1

**Files:**
- Create: `scripts/install_isaac6.sh` (new file; leave upstream `scripts/install.sh` alone)

**Interfaces:**
- Produces: `.venv/bin/python` with `isaacsim` 6.0.1, `torch` 2.11+cu128, GenManip `requirements.txt` (minus anything with no py3.12 wheel, reported).

- [x] **Step 1:** `uv venv --python 3.12 .venv`
- [x] **Step 2:** `uv pip install --python .venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`
- [x] **Step 3:** `uv pip install --python .venv/bin/python "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow` (10+ GB, run in background, poll). The two extra flags are required; without them resolution fails on `mujoco-usd-converter` / `tinyobjloader`.
- [x] **Step 4:** `grep -v '^mplib' requirements.txt | uv pip install --python .venv/bin/python -r /dev/stdin`. mplib skipped (numpy downgrade to 1.26 breaks Isaac Sim; rejected at runtime anyway). Everything else installed.
- [x] **Step 5 (check):**
  ```bash
  OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -c "from isaacsim import SimulationApp; a=SimulationApp({'headless':True}); import omni.usd; print('OK', omni.usd.get_context().get_stage()); a.close()"
  .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
  ```
  Expected: `OK Usd.Stage…` and `2.11.0+cu128 True (12, 0)`.
- [x] **Step 6:** Write `scripts/install_isaac6.sh` containing exactly the commands that worked (steps 1–4 + `export OMNI_KIT_ACCEPT_EULA=YES`), then commit: `git add scripts/install_isaac6.sh && git commit -m "port: Isaac Sim 6.0.1 install script (py3.12, torch cu128)"`.

---

### Task 2: cuRobo for sm_120

**Files:**
- Create: `saved/envs/curobo/` (gitignored clone)
- Modify: `scripts/install_isaac6.sh` (append the cuRobo section)

**Interfaces:**
- Consumes: Task 1 venv.
- Produces: `from curobo.wrap.reacher.motion_gen import MotionGen` importable; kernels run on the GPU.

- [x] **Step 1:** `mkdir -p saved/envs && git clone --depth 1 --branch v0.7.8 https://github.com/NVlabs/curobo.git saved/envs/curobo` (cuRobo `main` is v2 with a new API; GenManip imports the v1 `curobo.wrap`/`curobo.types` modules, so pin `v0.7.8`).
- [x] **Step 2:** Get an sm_120-capable nvcc without sudo. The pip wheels `nvidia-cuda-nvcc-cu12` (PyPI and pypi.nvidia.com) ship only `ptxas`/nvvm, no `nvcc` binary, so use conda packages via user-space micromamba instead: `micromamba create -p .venv/cuda-12.8 -c nvidia -c conda-forge --override-channels cuda-nvcc-impl=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8`, then symlink `include -> targets/x86_64-linux/include` and `lib64 -> lib` (see `scripts/install_isaac6.sh`). Original pip idea, kept for reference: `uv pip install --python .venv/bin/python "nvidia-cuda-nvcc-cu12==12.8.*" "nvidia-cuda-runtime-cu12==12.8.*" "nvidia-cuda-cccl-cu12==12.8.*"` then `export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvcc` and `export PATH=$CUDA_HOME/bin:$PATH`. Check `nvcc --version` prints 12.8.x. If the wheel layout lacks headers `cuda_runtime.h`, symlink `nvidia/cuda_runtime/include/*` into `$CUDA_HOME/include`.
- [x] **Step 3:** `export CUDA_HOME=$PWD/.venv/cuda-12.8 PATH=$CUDA_HOME/bin:$PATH TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8; uv pip install --python .venv/bin/python -e saved/envs/curobo --no-build-isolation` (build takes 5–15 min).
- [x] **Step 4 (check):**
  ```bash
  .venv/bin/python -c "
  from curobo.types.base import TensorDeviceType
  from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
  from curobo.types.robot import RobotConfig
  from curobo.util_file import get_robot_configs_path, join_path, load_yaml
  from curobo.types.math import Pose
  t=TensorDeviceType(); cfg=load_yaml(join_path(get_robot_configs_path(),'franka.yml'))['robot_cfg']
  ik=IKSolver(IKSolverConfig.load_from_robot_config(RobotConfig.from_dict(cfg,t),None,rotation_threshold=0.05,position_threshold=0.005,num_seeds=20,self_collision_check=False,self_collision_opt=False,tensor_args=t))
  q=ik.sample_configs(1); kin=ik.fk(q); r=ik.solve_batch(Pose(kin.ee_position,kin.ee_quaternion)); print('IK success', bool(r.success.all()))"
  ```
  Expected: `IK success True` with no `no kernel image` error.
- [x] **Step 5:** Append the working commands to `scripts/install_isaac6.sh`; commit `git commit -am "port: build cuRobo for sm_120 with pip nvcc 12.8"`.
- Fallback if pip nvcc cannot build: report to the user that a system CUDA 12.8 toolkit (runfile, toolkit only, sudo) is required; do not attempt sudo.

---

### Task 3: Mechanical import rename

**Files:**
- Modify: the 43 `.py` files listed by `grep -rl "omni\.isaac" --include=*.py .`

**Interfaces:**
- Produces: no `omni.isaac` import anywhere; all names imported under their old local aliases so call sites are untouched.

- [ ] **Step 1:** Apply, from `GenManip/`:
  ```bash
  grep -rl "omni\.isaac" --include=*.py . | xargs sed -i \
    -e 's/from omni\.isaac\.core import World/from isaacsim.core.api import World/' \
    -e 's/from omni\.isaac\.core\.robots\.robot import Robot/from isaacsim.core.api.robots.robot import Robot/' \
    -e 's/from omni\.isaac\.core\.articulations import Articulation/from isaacsim.core.prims import SingleArticulation as Articulation/' \
    -e 's/from omni\.isaac\.core\.prims import XFormPrim, RigidPrim/from isaacsim.core.prims import SingleXFormPrim as XFormPrim, SingleRigidPrim as RigidPrim/' \
    -e 's/from omni\.isaac\.core\.prims import XFormPrim/from isaacsim.core.prims import SingleXFormPrim as XFormPrim/' \
    -e 's/from omni\.isaac\.core\.prims import GeometryPrim/from isaacsim.core.prims import SingleGeometryPrim as GeometryPrim/' \
    -e 's/from omni\.isaac\.core\.materials\.omni_pbr import/from isaacsim.core.api.materials.omni_pbr import/' \
    -e 's/from omni\.isaac\.core\.materials import/from isaacsim.core.api.materials import/' \
    -e 's/from omni\.isaac\.core\.objects import/from isaacsim.core.api.objects import/' \
    -e 's/from omni\.isaac\.core\.utils\./from isaacsim.core.utils./' \
    -e 's/from omni\.isaac\.sensor import Camera/from isaacsim.sensors.camera import Camera/' \
    -e 's/from omni\.isaac\.franka import Franka/from isaacsim.robot.manipulators.examples.franka import Franka/'
  ```
- [ ] **Step 2 (check):** `grep -rn "omni\.isaac" --include=*.py . | grep -v "^\s*#"`; Expected: empty. Then `.venv/bin/python -m compileall -q genmanip standalone_tools demos demogen.py render.py`; Expected: no output.
- [x] **Step 3:** Also guard the mplib import if Task 1 reported no py3.12 wheel: in `genmanip/core/robot/base.py`, `genmanip/utils/loader/scene.py`, `genmanip/utils/loader/domain_randomization.py` wrap the `mplib` imports in `try/except ImportError: MplibPlanner = None` (and `Pose = None`). Skip if mplib installed.
- [ ] **Step 4:** `git add -A genmanip standalone_tools demos && git commit -m "port: rename omni.isaac.* imports to isaacsim.* (Isaac Sim 4.5+ names)"`

---

### Task 4: Runtime import walk

**Files:**
- Create: `standalone_tools/verify/check_imports_isaac6.py`

**Interfaces:**
- Consumes: Task 1 venv, Task 2 cuRobo, Task 3 renames.
- Produces: a script that imports every `genmanip` module inside a headless `SimulationApp` and fails on the first `ImportError`; a list of deprecation warnings.

- [x] **Step 1:** Write the script:
  ```python
  import os, sys, pkgutil, importlib, warnings, traceback
  os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
  from isaacsim import SimulationApp
  app = SimulationApp({"headless": True})
  sys.path.insert(0, os.getcwd())
  import genmanip
  failed, warned = [], set()
  with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      for m in pkgutil.walk_packages(genmanip.__path__, "genmanip."):
          try:
              importlib.import_module(m.name)
          except Exception as e:  # report all, keep going
              failed.append((m.name, f"{type(e).__name__}: {e}"))
      warned = {str(x.message)[:200] for x in w if "deprecat" in str(x.message).lower()}
  app.close()
  print("DEPRECATIONS:", *sorted(warned), sep="\n  ")
  print("FAILED:", *failed, sep="\n  ")
  sys.exit(1 if failed else 0)
  ```
- [ ] **Step 2 (check):** `.venv/bin/python standalone_tools/verify/check_imports_isaac6.py` (2026-09-10: 99 modules; all non-curobo name-level failures fixed; remaining FAILED = curobo (pending Task 2), roboticstoolbox-python 1.1.1 wheel built against numpy 1.x (`numpy.core.multiarray failed to import`; fix: `uv pip install --python .venv/bin/python --no-binary roboticstoolbox-python --reinstall-package roboticstoolbox-python roboticstoolbox-python`), `gpt_utils` needs `OPENAI_API_KEY`, `io_utils` needs system `libturbojpeg`. With those three stubbed every module imports.); Expected: `FAILED:` with nothing after it, exit 0. Fix each failure in place (wrong class name, moved module) using `dir()` of the target package; only names, no logic changes.
- [x] **Step 3:** Record the deprecation list in `plans/2026-09-10-isaac-6-port-plan.md` under "Deprecations seen" (below).
- [x] **Step 4:** `git add standalone_tools/verify/check_imports_isaac6.py genmanip && git commit -m "port: import-walk check for Isaac Sim 6; fix moved names"`

---

### Task 5: Baseline demogen (Franka + Robotiq banana)

**Files:**
- Create: `saved/configs/banana_smoke.yml` (copy of `configs/tasks/banana_robotiq_example.yml` with `num_episode: 2`; gitignored)
- Modify: whatever runtime breaks, minimal diffs.

**Interfaces:**
- Consumes: assets in `saved/assets/` (from `standalone_tools/download_assets.py`).
- Produces: a trajectory dir under `saved/` whose `log.txt` contains `success`.

- [ ] **Step 1:** Confirm assets: `ls saved/assets/` shows the banana scene and object usds; if not, run `.venv/bin/python standalone_tools/download_assets.py`.
- [ ] **Step 2:** `cp configs/tasks/banana_robotiq_example.yml saved/configs/banana_smoke.yml && sed -i 's/num_episode: 100/num_episode: 2/' saved/configs/banana_smoke.yml`
- [ ] **Step 3 (run):** `OMNI_KIT_ACCEPT_EULA=YES timeout 3600 .venv/bin/python demogen.py -cfg saved/configs/banana_smoke.yml 2>&1 | tee saved/demogen_smoke.log`
- [ ] **Step 4 (check):** `grep -rl success saved/**/log.txt | head`; Expected: ≥1 path. If the run crashes, fix the first traceback with the smallest change (typical: private Camera attrs, `set_solver_*` on `SingleArticulation`, semantics API) and rerun. Log each fix as a bullet under "Runtime fixes" below.
- [ ] **Step 5:** `.venv/bin/python standalone_tools/verify/check_data_valid.py --data_path <trajectory dir> --type trajectory`; Expected: valid.
- [ ] **Step 6:** Commit fixes: `git commit -am "port: runtime fixes for demogen on Isaac Sim 6.0"`.

---

### Task 6: Baseline render + visual check

**Files:**
- Modify: whatever breaks in `genmanip/demogen/workflow/render.py` / `render_recorder.py`, minimal.

**Interfaces:**
- Consumes: Task 5 trajectory.
- Produces: rendered dataset with RGB per camera, an mp4 and a PNG frame under `saved/vis/`.

- [ ] **Step 1 (run):** `OMNI_KIT_ACCEPT_EULA=YES timeout 3600 .venv/bin/python render.py -cfg saved/configs/banana_smoke.yml 2>&1 | tee saved/render_smoke.log`
- [ ] **Step 2 (check):** `.venv/bin/python standalone_tools/data_vis.py --data_path <episode> --camera_name <first camera in configs/cameras/fixed_camera_robotiq.yml> --output_path saved/vis/banana.mp4`; then extract a frame: `ffmpeg -y -i saved/vis/banana.mp4 -vf "select=eq(n\,30)" -vframes 1 saved/vis/frame30.png` and check mean brightness is in a sane range: `.venv/bin/python -c "import cv2;i=cv2.imread('saved/vis/frame30.png');print(i.shape,i.mean())"`; Expected: shape `(H,W,3)`, mean between 30 and 220. Read the PNG and confirm the robot, table and banana are visible.
- [ ] **Step 3:** Print the active render mode for the record: inside a `SimulationApp`, `import carb; print(carb.settings.get_settings().get('/rtx/rendermode'))`. Note it in "Runtime fixes".
- [ ] **Step 4:** Commit fixes: `git commit -am "port: render path on Isaac Sim 6.0 (RTX Real-Time 2.0)"`.

---

### Task 7: Docs and pins

**Files:**
- Modify: `/home/adnan/rena/rena-sim/CLAUDE.md` "Hard constraints" and "Commands" sections
- Modify: `README.md` (one short "Isaac Sim 6.0" install note pointing at `scripts/install_isaac6.sh`)

- [ ] **Step 1:** Replace the 4.1.0/py3.10 constraint with: Isaac Sim 6.0.1 pip, Python 3.12, torch 2.11 cu128, PhysX, RTX Real-Time 2.0 default, RTX 5090 (sm_120) requires this; Newton experimental, not used. Replace `bash scripts/install.sh` with `bash scripts/install_isaac6.sh`.
- [ ] **Step 2:** `git -C GenManip commit -am "port: README note for Isaac Sim 6.0 install"`; push: `git push -u origin port/isaac-6.0`.

---

### Task 8: Skills from what actually worked

Spawn one `skill-creator:skill-creator` agent per skill after the underlying task passed, giving it the exact commands and error/fix pairs:

- [ ] `isaac-sim-pip-blackwell-env`: Task 1 + Task 2 (venv, cu128 torch, pip nvcc for cuRobo on sm_120, EULA, smoke launch).
- [ ] `isaac-4x-to-6-import-port`: Task 3 + Task 4 (rename map, import-walk check, the runtime fixes list).
- [ ] `genmanip-baseline-smoke`: Task 5 + Task 6 (2-episode config, demogen → render → validate → frame check, where logs land).

---

## Deprecations seen

Import walk 2026-09-10 (Isaac Sim 6.0.1, 99 `genmanip` modules):

- Python `warnings` with "deprecat": none.
- Kit log: `[Warning] [pxr.Semantics] pxr.Semantics is deprecated - please use Semantics instead` (from Isaac's own `isaacsim.core.utils.semantics`, not our code).
- `isaacsim.core.utils.semantics.add_update_semantics` is gone; ported `genmanip/utils/usd_utils/semantic_utils.py` to `add_labels(prim, [label], instance_name="class")` (LabelsAPI instance "class").
- `isaacsim.robot.manipulators.examples` is not auto-enabled; `enable_extension("isaacsim.robot.manipulators.examples")` added before the Franka import in `panda_hand.py` and `planner/mplib/utils.py`.
- Python 3.12 dataclasses reject `np.ndarray` defaults (`mutable default ... use default_factory`); fixed in `genmanip/utils/pointcloud/utils.py`.
- `SimulationApp.close()` with `fastShutdown` calls `os._exit()`; print results and pass `exit_code=` to `close()` instead of `sys.exit` after it.

## Runtime fixes

(filled by Tasks 5–6)

## Adversarial review 2026-09-10 (gpt-6-astra) — verdicts after checking source

1. Confirmed: `pkgutil.walk_packages` yields 0 modules (namespace packages). Fixed in `check_imports_isaac6.py` (glob-based, asserts count).
2. Confirmed: stock banana task requires the private AnyGrasp server (`configs/miscs/anygrasp.yml` → 10.150.123.24, POST without timeout). **Baseline switched to `configs/tasks/minimal.yml`** (`force_fixed_grasp: true` → local top-down grasp in `get_init_grasp`). Explicit baseline change.
3. Confirmed: success is counted in `trajectory/log.pkl` (`record_log`), no `log.txt`. Gate = `log.pkl["success"]` increases and new episode dirs appear.
4. Confirmed: `download_assets.py --dataset` is required (done: basic, banana, banana-layout, franka-robotiq); `saved/configs/` must be created.
5. Confirmed: demogen skips a task whose `log.pkl` already has ≥ num_episode successes; render skips existing episode dirs. Smoke run uses a unique `task_name` and `set -o pipefail`.
6. Confirmed: `check_data_valid.py` prints but exits 0; validates keys not RGB. Task 6 decodes RGB for every camera in the camera config and compares frame counts.
7. Confirmed: `genmanip/utils/planner/mplib/utils.py` also needed guarding. Done in `33a6096`.
