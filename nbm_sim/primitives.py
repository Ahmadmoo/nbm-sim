"""Motion primitives: the action set for starts and continuations (option C).

Every primitive has the same duration and starts and ends at rest, so any primitive can follow any
other and all cost the same time. Twists are body-frame (optical) and are built from the current
pose and the declared target-center prior:

| kind     | motion                                                        | parallax |
|----------|---------------------------------------------------------------|----------|
| orbit    | translate in the image plane, rotate to keep the target fixed | yes      |
| slide    | translate in the image plane, no rotation                     | yes      |
| radial   | move toward / away from the target                            | weak     |
| rotate   | rotate only (same image motion of the target as an orbit)     | none     |
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Primitive:
    name: str
    kind: str
    angle_deg: float = 0.0          # image-plane direction: 0 = right (+x), 90 = down (+y)
    sign: int = 1                   # radial: +1 toward the target, -1 away


def make_library(orbit_dirs=8, slide_dirs=4, rotate_dirs=4, radial=True):
    lib = [Primitive(f"orbit_{a:g}", "orbit", a) for a in np.arange(orbit_dirs) * 360.0 / orbit_dirs]
    lib += [Primitive(f"slide_{a:g}", "slide", a) for a in np.arange(slide_dirs) * 360.0 / slide_dirs]
    if radial:
        lib += [Primitive("approach", "radial", sign=1), Primitive("retreat", "radial", sign=-1)]
    lib += [Primitive(f"rotate_{a:g}", "rotate", a) for a in np.arange(rotate_dirs) * 360.0 / rotate_dirs]
    return lib


def instantiate(prim, camera_state, cfg, speed=0.12, move_s=0.6, stop_s=0.4):
    """``[(twist, duration), ...]`` in whole action steps: constant twist, then zeros to come to rest."""
    T, c = camera_state["T_wc"], np.asarray(camera_state["target_center_prior"], np.float64)
    d = max(float((c - T[:3, 3]) @ T[:3, 2]), 0.2)        # distance to the target along the optical axis
    a = math.radians(prim.angle_deg)
    ux, uy = math.cos(a), math.sin(a)
    if prim.kind == "orbit":            # rotation axis passes through the point d in front of the camera
        tw = [speed * ux, speed * uy, 0.0, speed * uy / d, -speed * ux / d, 0.0]
    elif prim.kind == "slide":
        tw = [speed * ux, speed * uy, 0.0, 0.0, 0.0, 0.0]
    elif prim.kind == "radial":
        tw = [0.0, 0.0, prim.sign * speed, 0.0, 0.0, 0.0]
    elif prim.kind == "rotate":
        tw = [0.0, 0.0, 0.0, speed * uy / d, -speed * ux / d, 0.0]
    else:
        raise ValueError(f"unknown primitive kind {prim.kind!r}")
    n_move, n_stop = round(move_s / cfg.action_dt), round(stop_s / cfg.action_dt)
    if n_move < 1 or abs(n_move * cfg.action_dt - move_s) > 1e-9 or abs(n_stop * cfg.action_dt - stop_s) > 1e-9:
        raise ValueError("move_s and stop_s must be whole multiples of action_dt")
    return [(np.array(tw), cfg.action_dt)] * n_move + [(np.zeros(6), cfg.action_dt)] * n_stop


def predict_path(env, schedule, sample_every=0.05):
    """Run ``schedule`` on a copy of the env's motion controller (same limits and safety checks,
    no rendering). Returns sampled poses, feasibility and whether it ends at rest."""
    ctrl = copy.deepcopy(env.motion)
    every = env.cfg.ticks(sample_every)
    poses, tick = [ctrl.T_wc.copy()], 0
    for v, dur in schedule:
        cmd, _ = ctrl.limit(np.asarray(v, np.float64))
        for _ in range(env.cfg.ticks(dur)):
            r = ctrl.tick(cmd, env.dt)
            if not r.accepted:
                return dict(feasible=False, reason=r.reason, poses=np.stack(poses), at_rest=False)
            tick += 1
            if tick % every == 0:
                poses.append(r.T_wc)
    at_rest = bool(np.linalg.norm(ctrl.v_w) < 1e-9 and np.linalg.norm(ctrl.w_w) < 1e-9)
    return dict(feasible=True, reason="", poses=np.stack(poses), at_rest=at_rest,
                path_length=ctrl.path_length - env.motion.path_length,
                rotation=ctrl.rotation_travel - env.motion.rotation_travel)
