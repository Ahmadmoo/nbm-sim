"""Six-axis body-twist execution with norm limits, world-frame slew limits and swept safety checks (spec §6)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import se3_exp


@dataclass(frozen=True)
class Box:
    """Axis-aligned box collider."""
    name: str
    center: tuple
    size: tuple

    def sdf(self, p):
        q = np.abs(np.asarray(p) - np.asarray(self.center)) - np.asarray(self.size) / 2.0
        return float(np.linalg.norm(np.maximum(q, 0.0)) + min(q.max(), 0.0))


@dataclass(frozen=True)
class Workspace:
    """Cylindrical shell around ``center_xy`` for the optical center."""
    center_xy: tuple
    radius: tuple
    height: tuple

    def margin(self, p):
        rho = float(np.hypot(p[0] - self.center_xy[0], p[1] - self.center_xy[1]))
        return min(rho - self.radius[0], self.radius[1] - rho, p[2] - self.height[0], self.height[1] - p[2])


@dataclass
class TickResult:
    accepted: bool
    T_wc: np.ndarray
    twist: np.ndarray
    reason: str = ""


class MotionController:
    """Integrates ``T_wc <- T_wc Exp(dt * hat(xi_body))`` one base tick at a time.

    Acceleration limits act on world-frame velocity at tick boundaries. Within a tick the
    body twist is constant, so world angular velocity is constant and world linear velocity
    rotates by at most ``|w| |v| dt``; that rotation is charged against the linear slew budget.
    """

    def __init__(self, cfg, colliders, workspace: Workspace):
        self.vmax, self.wmax = cfg.max_linear_speed, cfg.max_angular_speed
        self.amax, self.alphamax = cfg.max_linear_acceleration, cfg.max_angular_acceleration
        self.clearance = cfg.camera_radius + cfg.collision_clearance
        self.colliders = list(colliders)
        self.workspace = workspace
        self.T_wc = np.eye(4)
        self.v_w = np.zeros(3)
        self.w_w = np.zeros(3)
        self.path_length = 0.0
        self.rotation_travel = 0.0

    def reset(self, T_wc):
        reason = self.check_pose(T_wc[:3, 3])
        if reason:
            raise ValueError(f"initial pose is not safe: {reason}")
        self.T_wc = np.array(T_wc, dtype=np.float64)
        self.v_w[:] = 0.0
        self.w_w[:] = 0.0
        self.path_length = 0.0
        self.rotation_travel = 0.0

    def check_pose(self, p, sweep=0.0):
        if self.workspace.margin(p) - sweep < 0.0:
            return "workspace_violation"
        for c in self.colliders:
            if c.sdf(p) - sweep < self.clearance:
                return "collision_attempt"
        return ""

    def limit(self, cmd):
        v, w = np.array(cmd[:3], dtype=np.float64), np.array(cmd[3:], dtype=np.float64)
        nv, nw = np.linalg.norm(v), np.linalg.norm(w)
        limited = nv > self.vmax or nw > self.wmax
        if nv > self.vmax:
            v *= self.vmax / nv
        if nw > self.wmax:
            w *= self.wmax / nw
        return np.concatenate([v, w]), bool(limited)

    def propose(self, target_body, dt):
        R = self.T_wc[:3, :3]
        w_tgt, v_tgt = R @ target_body[3:], R @ target_body[:3]
        dw = w_tgt - self.w_w
        n = np.linalg.norm(dw)
        if n > self.alphamax * dt:
            dw *= self.alphamax * dt / n
        w_new = self.w_w + dw
        budget = max(0.0, self.amax * dt - np.linalg.norm(w_new) * max(np.linalg.norm(self.v_w),
                                                                        np.linalg.norm(v_tgt)) * dt)
        dv = v_tgt - self.v_w
        n = np.linalg.norm(dv)
        if n > budget:
            dv = dv * (budget / n) if n > 0 else dv
        v_new = self.v_w + dv
        twist = np.concatenate([R.T @ v_new, R.T @ w_new])
        T_new = self.T_wc @ se3_exp(dt * twist)
        return twist, T_new

    def tick(self, target_body, dt):
        twist, T_new = self.propose(target_body, dt)
        half_sweep = 0.5 * np.linalg.norm(twist[:3]) * dt
        p0, p1 = self.T_wc[:3, 3], T_new[:3, 3]
        reason = self.check_pose(p0, half_sweep) or self.check_pose(p1, half_sweep)
        if reason:
            return TickResult(False, self.T_wc.copy(), np.zeros(6), reason)
        self.T_wc = T_new
        self.v_w = T_new[:3, :3] @ twist[:3]
        self.w_w = T_new[:3, :3] @ twist[3:]
        self.path_length += float(np.linalg.norm(twist[:3]) * dt)
        self.rotation_travel += float(np.linalg.norm(twist[3:]) * dt)
        return TickResult(True, T_new.copy(), twist)

    def emergency_stop(self):
        """Idealized stop at the last accepted safe state (reported, not normal execution)."""
        self.v_w[:] = 0.0
        self.w_w[:] = 0.0

    def get_state(self):
        return dict(T_wc=self.T_wc.copy(), v_w=self.v_w.copy(), w_w=self.w_w.copy(),
                    path_length=self.path_length, rotation_travel=self.rotation_travel)

    def set_state(self, s):
        self.T_wc = s["T_wc"].copy()
        self.v_w = s["v_w"].copy()
        self.w_w = s["w_w"].copy()
        self.path_length = s["path_length"]
        self.rotation_travel = s["rotation_travel"]
