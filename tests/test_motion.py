import numpy as np

from nbm_sim.config import SimConfig
from nbm_sim.evaluation import check_motion_limits
from nbm_sim.geometry import look_at_T_wc, se3_exp
from nbm_sim.motion import Box, MotionController, Workspace

WS = Workspace((0.0, 0.0), (0.4, 1.2), (0.9, 1.6))


def run(ctrl, cmd, n, dt):
    poses, tw = [ctrl.T_wc.copy()], []
    for _ in range(n):
        r = ctrl.tick(cmd, dt)
        if not r.accepted:
            return poses, tw, r.reason
        poses.append(r.T_wc)
        tw.append(r.twist)
    return poses, tw, ""


def test_unbounded_constant_twist_matches_exp():
    cfg = SimConfig(max_linear_acceleration=np.inf, max_angular_acceleration=np.inf, max_linear_speed=10,
                    max_angular_speed=10)
    c = MotionController(cfg, [], Workspace((0, 0), (0, 100), (-100, 100)))
    T0 = look_at_T_wc((0.8, 0, 1.2), (0, 0, 1.0))
    c.reset(T0)
    xi = np.array([0.1, -0.05, 0.08, 0.2, -0.4, 0.3])
    poses, _, _ = run(c, xi, 500, 0.001)
    Tref = T0 @ se3_exp(0.5 * xi)
    assert np.abs(poses[-1] - Tref).max() < 1e-6


def test_limits_respected_under_random_commands():
    cfg = SimConfig()
    c = MotionController(cfg, [], Workspace((0, 0), (0, 100), (-100, 100)))
    c.reset(look_at_T_wc((0.8, 0, 1.2), (0, 0, 1.0)))
    rng = np.random.default_rng(0)
    poses, tws = [c.T_wc.copy()], []
    for _ in range(60):
        cmd, _ = c.limit(rng.normal(size=6) * np.r_[0.4, 0.4, 0.4, 1, 1, 1])
        p, t, _ = run(c, cmd, 50, cfg.base_dt)
        poses += p[1:]
        tws += t
    res = check_motion_limits(np.array(tws), np.array(poses), cfg.base_dt, cfg)
    assert res["ok"], res


def test_norm_limit_flags():
    c = MotionController(SimConfig(), [], WS)
    out, flag = c.limit(np.array([1.0, 0, 0, 0, 0, 2.0]))
    assert flag and np.isclose(np.linalg.norm(out[:3]), 0.2) and np.isclose(np.linalg.norm(out[3:]), 0.5)
    out, flag = c.limit(np.array([0.1, 0, 0, 0, 0, 0.1]))
    assert not flag


def test_zero_command_decelerates_to_rest():
    cfg = SimConfig()
    c = MotionController(cfg, [], Workspace((0, 0), (0, 100), (-100, 100)))
    c.reset(look_at_T_wc((0.8, 0, 1.2), (0, 0, 1.0)))
    run(c, np.array([0.2, 0, 0, 0, 0, 0.5]), 800, cfg.base_dt)
    run(c, np.zeros(6), 800, cfg.base_dt)
    assert np.linalg.norm(c.v_w) < 1e-12 and np.linalg.norm(c.w_w) < 1e-12


def test_workspace_violation_stops_at_safe_state():
    cfg = SimConfig(max_linear_acceleration=np.inf)
    c = MotionController(cfg, [], WS)
    c.reset(look_at_T_wc((1.15, 0, 1.2), (0, 0, 1.0)))
    poses, _, reason = run(c, np.array([0, 0, -0.2, 0, 0, 0]), 1000, cfg.base_dt)  # optical -z = backward
    assert reason == "workspace_violation"
    assert WS.margin(poses[-1][:3, 3]) >= 0


def test_collision_detected():
    cfg = SimConfig(max_linear_acceleration=np.inf)
    box = Box("obstacle", (0.7, 0.0, 1.2), (0.1, 0.4, 0.4))
    c = MotionController(cfg, [box], WS)
    c.reset(look_at_T_wc((1.0, 0, 1.2), (0, 0, 1.2)))
    poses, _, reason = run(c, np.array([0, 0, 0.2, 0, 0, 0]), 3000, cfg.base_dt)
    assert reason == "collision_attempt"
    assert box.sdf(poses[-1][:3, 3]) >= cfg.camera_radius + cfg.collision_clearance


def test_path_accounting():
    cfg = SimConfig(max_linear_acceleration=np.inf, max_angular_acceleration=np.inf)
    c = MotionController(cfg, [], Workspace((0, 0), (0, 100), (-100, 100)))
    c.reset(look_at_T_wc((0.8, 0, 1.2), (0, 0, 1.0)))
    run(c, np.array([0.1, 0, 0, 0, 0.2, 0]), 1000, cfg.base_dt)
    assert np.isclose(c.path_length, 0.1) and np.isclose(c.rotation_travel, 0.2)
