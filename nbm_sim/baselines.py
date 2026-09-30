"""Bootstrap schedule and fixed-motion baselines (spec §13). All go through the same
controller, limits and safety checks as any planner."""
from __future__ import annotations

import numpy as np

from .geometry import look_at_R_wc, so3_log


def bootstrap_lateral_scan(cfg, speed=None):
    """1.0 s continuous lateral scan along camera +X that ends at rest (saved physical-twist schedule).

    Accelerates to ``speed``, cruises, then commands zero so the controller decelerates to rest
    within the bootstrap window. Durations are whole action steps.
    """
    speed = cfg.max_linear_speed if speed is None else speed
    n_total = round(cfg.bootstrap_seconds / cfg.action_dt)
    n_stop = int(np.ceil(speed / cfg.max_linear_acceleration / cfg.action_dt))
    if n_stop >= n_total:
        raise ValueError("bootstrap too short to stop from the requested speed")
    move = [(np.array([speed, 0, 0, 0, 0, 0.0]), cfg.action_dt)] * (n_total - n_stop)
    return move + [(np.zeros(6), cfg.action_dt)] * n_stop


def _look_at_correction(state, target, gain):
    R = state["T_wc"][:3, :3]
    R_des = look_at_R_wc(state["T_wc"][:3, 3], target)
    return gain * so3_log(R.T @ R_des)   # body-frame rotation vector toward the look-at orientation


class Baseline:
    """Planner interface: ``act(map_state, uncertainty, observation_history, camera_state) -> velocity``."""
    uses_target_prior = False

    def act(self, map_state=None, uncertainty=None, observation_history=None, camera_state=None):
        raise NotImplementedError


class FixedOrbit(Baseline):
    """Constant-speed orbit about the target's vertical axis at the current radius/height with a
    scripted look-at orientation. Uses the known target center as a declared prior."""
    uses_target_prior = True

    def __init__(self, speed=0.15, direction=1, look_gain=2.0, radial_gain=1.0, height_gain=1.0):
        self.speed, self.dir, self.kl, self.kr, self.kh = speed, direction, look_gain, radial_gain, height_gain
        self.r0 = self.z0 = None

    def act(self, map_state=None, uncertainty=None, observation_history=None, camera_state=None):
        s = camera_state
        c, p = s["target_center_prior"], s["T_wc"][:3, 3]
        d = p[:2] - c[:2]
        r = np.linalg.norm(d)
        if self.r0 is None:
            self.r0, self.z0 = r, p[2]
        radial = d / r
        tangent = self.dir * np.array([-radial[1], radial[0]])
        omega = self.dir * self.speed / r
        v_w = np.r_[self.speed * tangent + self.kr * (self.r0 - r) * radial, self.kh * (self.z0 - p[2])]
        R = s["T_wc"][:3, :3]
        w_b = R.T @ np.array([0, 0, omega]) + _look_at_correction(s, c, self.kl)
        return np.r_[R.T @ v_w, w_b]


class FixedScan(Baseline):
    """Predefined body-frame translations with lateral and vertical parallax, with look-at correction."""
    uses_target_prior = True

    def __init__(self, speed=0.12, segment_seconds=1.5, look_gain=2.0):
        self.dirs = [np.array(d, float) for d in ([1, 0, 0], [0, -1, 0], [-1, 0, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0])]
        self.speed, self.seg, self.kl = speed, segment_seconds, look_gain
        self.t0 = None

    def act(self, map_state=None, uncertainty=None, observation_history=None, camera_state=None):
        s = camera_state
        self.t0 = s["t"] if self.t0 is None else self.t0
        k = int((s["t"] - self.t0) // self.seg) % len(self.dirs)
        return np.r_[self.speed * self.dirs[k], _look_at_correction(s, s["target_center_prior"], self.kl)]


class BoundedRandom(Baseline):
    """Smooth random body-velocity targets held for ``hold_seconds``, pulled back toward the workspace
    middle near its boundary, with optional look-at correction."""
    uses_target_prior = True

    def __init__(self, rng, cfg, hold_seconds=0.5, scale=0.6, look_gain=1.5, boundary_margin=0.08):
        self.rng, self.cfg, self.hold, self.scale, self.kl, self.m = rng, cfg, hold_seconds, scale, look_gain, boundary_margin
        self.next_t, self.v = -np.inf, np.zeros(6)

    def _ball(self, n, radius):
        d = self.rng.normal(size=n)
        return d / np.linalg.norm(d) * radius * self.rng.random() ** (1 / n)

    def act(self, map_state=None, uncertainty=None, observation_history=None, camera_state=None):
        s, c = camera_state, self.cfg
        if s["t"] >= self.next_t:
            self.v = np.r_[self._ball(3, self.scale * c.max_linear_speed), self._ball(3, 0.3 * c.max_angular_speed)]
            self.next_t = s["t"] + self.hold
        R, p = s["T_wc"][:3, :3], s["T_wc"][:3, 3]
        ctr = s["target_center_prior"]
        d = p[:2] - ctr[:2]
        r = np.linalg.norm(d)
        (r0, r1), (z0, z1) = c.horizontal_workspace_radius, c.camera_height
        push = np.zeros(3)
        if r - r0 < self.m:
            push[:2] += d / r
        if r1 - r < self.m:
            push[:2] -= d / r
        if p[2] - z0 < self.m:
            push[2] += 1
        if z1 - p[2] < self.m:
            push[2] -= 1
        v = self.v.copy()
        if push.any():
            v[:3] = R.T @ (push / np.linalg.norm(push) * self.scale * c.max_linear_speed)
        if self.kl:
            v[3:] = _look_at_correction(s, ctr, self.kl) + 0.3 * v[3:]
        return v


def run_planner(env, planner, n_steps=None, **context):
    """Drive ``planner`` until the episode ends or ``n_steps`` policy steps; returns public packets."""
    packets = []
    while n_steps is None or len(packets) < n_steps:
        v = planner.act(camera_state=env.camera_state(), **context)
        pkt = env.step(v)
        packets.append(pkt)
        if pkt["terminated"] or pkt["truncated"]:
            break
    return packets
