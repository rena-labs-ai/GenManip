# GenManip → Isaac Sim 6.0.1 (PhysX) port — design

Date: 2026-09-10. Status: approved in chat (direction: 6.0.x on PhysX, not Newton).

## Why

- Hardware is an RTX 5090 Laptop (Blackwell, sm_120), driver 580.178.04. The pinned stack
  (Isaac Sim 4.1.0, torch 2.4 cu121, CUDA 12.1) predates Blackwell and cannot run here.
- Isaac Sim 6.1 does not exist (latest 6.0.1, 2026-06-22). Newton in 6.0 is experimental,
  MJWarp-only, tested on humanoids/hands, and ignores every PhysX knob GenManip uses to
  stabilize grasps (solver iteration counts, contact/rest offsets, armature, max
  depenetration, SDF/convex-decomposition colliders). Rendering is unaffected by the
  physics backend, so Newton buys nothing for visual fidelity.
- 6.0 removed the `omni.isaac.*` shims but keeps `isaacsim.core.api/prims/utils` and
  `isaacsim.sensors.camera` as deprecated-but-functional. The port is a rename plus a
  toolchain bump, not a rewrite.

## Target stack

| Component | Value |
|---|---|
| Python | 3.12 (`uv venv --python 3.12 GenManip/.venv`) |
| torch | 2.11.0 cu128 (sm_120) from download.pytorch.org |
| Isaac Sim | `isaacsim[all,extscache]==6.0.1.0` from pypi.nvidia.com |
| Physics | PhysX (default). Newton not used. |
| Renderer | RTX Real-Time 2.0 (6.0 default). Accept the look change; validate vs real frames. |
| cuRobo | NVlabs/curobo tag `v0.7.8` (main is the v2 API; GenManip uses v1 `curobo.wrap`/`curobo.types`), built with nvcc 12.8 via user-space micromamba (sm_120) |
| mplib | keep pinned; guard the import so a missing wheel does not break curobo runs |

## Rename map (Isaac Sim 4.5 extension split, still valid in 6.0)

| Old | New |
|---|---|
| `from omni.isaac.core import World` | `from isaacsim.core.api import World` |
| `from omni.isaac.core.robots.robot import Robot` | `from isaacsim.core.api.robots.robot import Robot` |
| `from omni.isaac.core.articulations import Articulation` | `from isaacsim.core.prims import SingleArticulation as Articulation` |
| `from omni.isaac.core.prims import XFormPrim` | `from isaacsim.core.prims import SingleXFormPrim as XFormPrim` |
| `RigidPrim` / `GeometryPrim` (same module) | `SingleRigidPrim as RigidPrim` / `SingleGeometryPrim as GeometryPrim` |
| `omni.isaac.core.utils.{prims,stage,extensions,types,semantics}` | `isaacsim.core.utils.{same}` |
| `omni.isaac.core.materials.omni_pbr.OmniPBR`, `omni.isaac.core.materials.PhysicsMaterial` | `isaacsim.core.api.materials.…` |
| `omni.isaac.core.objects.VisualCuboid` | `isaacsim.core.api.objects.VisualCuboid` |
| `from omni.isaac.sensor import Camera` | `from isaacsim.sensors.camera import Camera` |
| `from omni.isaac.franka import Franka` | `from isaacsim.robot.manipulators.examples.franka import Franka` |

`from isaacsim import SimulationApp`, `omni.usd`, `omni.replicator.core`,
`omni.kit.asset_converter`, `omni.kit.usd.collect`, `pxr.*` are unchanged.

Known follow-ups the rename does not cover (verify at runtime, fix only if broken):
`add_update_semantics` may be deprecated for `add_labels` (UsdSemantics.LabelsAPI);
`Camera(frequency=…)`, `camera._custom_annotators`, `camera._render_product` are private
surfaces of the deprecated Camera; `set_solver_*_iteration_count` on `SingleArticulation`.

## Scope

In: environment, cuRobo, import rename, runtime fixes for the stock Franka banana task
(`configs/tasks/banana_robotiq_example.yml`) through `demogen.py` → `render.py` →
`check_data_valid.py` → `data_vis.py`, docs, install script, skills capturing the
repeatable workflows.

Out: Newton, the experimental `isaacsim.core.experimental` API, the Rena robot itself
(next plan), Ray/eval cluster paths, mplib planner.

## Success criteria

1. Headless `demogen.py` on the banana task produces ≥1 episode whose `log.txt` says `success`.
2. `render.py` on that trajectory writes RGB for every configured camera; `check_data_valid.py --type trajectory` passes; `data_vis.py` writes a viewable mp4 and a sampled frame is not black or washed out.
3. `grep -r "omni\.isaac" --include=*.py` finds nothing outside comments/docs.
4. Every `genmanip` module imports inside a `SimulationApp` (walk_packages check).
5. A fresh clone + `scripts/install_isaac6.sh` reproduces the environment.

## Working conventions

Branch `port/isaac-6.0` in `GenManip/` (no worktree: `.venv` and `saved/` live inside
the repo and are gitignored). Push to `origin` only. Keep upstream-touching diffs
mechanical (renames) so `git merge upstream/main` stays cheap; new files for anything
rena-specific.
