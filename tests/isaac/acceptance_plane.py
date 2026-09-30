# Isaac-side acceptance checks on the textured-plane scene (spec §17): intrinsics, depth,
# six-axis sign convention, and render-buffer latency. Run with the Isaac Sim Python:
#   python tests/isaac/acceptance_plane.py      (or ${ISAACLAB}/isaaclab.sh -p ...)

# %% Parameters
HEADLESS = True
OUT_DIR = "./acceptance"
LINEAR_STEP = 0.02      # m, per-axis probe
ANGULAR_STEP = 0.03     # rad, per-axis probe
JUMP = 0.05             # m, latency probe (test-only pose jump outside an episode)

# %% Launch Isaac and build the scene
import json
import pathlib

import numpy as np

from nbm_sim.camera import launch_app
from nbm_sim.config import SimConfig

cfg = SimConfig(scene_id="textured_plane", display=not HEADLESS, record=False)
app = launch_app(cfg)

from nbm_sim.environment import CameraNBMEnv
from nbm_sim.geometry import project, se3_exp

env = CameraNBMEnv(cfg)
env.reset()
T0 = env.motion.T_wc.copy()
landmarks = [b for b in env.spec.boxes if b.name.startswith("landmark")]
fronts = np.array([[b.center[0], b.center[1] - b.size[1] / 2, b.center[2]] for b in landmarks])
report = dict(scene="textured_plane", backend=env.backend.describe())


def centroids(T):
    sem = env.backend.capture(T, ("semantic",))["semantic"]
    out = []
    for b in landmarks:
        m = sem.get(b.semantic)
        ys, xs = np.nonzero(m) if m is not None else ([], [])
        out.append([np.mean(xs), np.mean(ys)] if len(xs) else [np.nan, np.nan])
    return np.array(out)


# %% Intrinsics: landmark projection within 0.5 px
meas = centroids(T0)
pred, _ = project(env.K, T0, fronts)
err = np.linalg.norm(meas - pred, axis=1)
report["intrinsics"] = dict(max_px_error=float(np.nanmax(err)), per_landmark=err.tolist(),
                            all_landmarks_seen=bool(np.isfinite(err).all()),
                            passed=bool(np.isfinite(err).all() and err.max() < 0.5))

# %% Depth: fronto-parallel plane at optical Z = 1.0 m within 1 mm, center and corners
out = env.backend.capture(T0, ("depth", "semantic"))
Z = out["depth"]
lm = np.zeros_like(out["depth_valid"])
for b in landmarks:
    lm |= out["semantic"].get(b.semantic, np.zeros_like(lm))
H, W = Z.shape
regions = dict(center=(H // 2, W // 2), top_left=(12, 12), top_right=(12, W - 13), bottom_left=(H - 13, 12),
               bottom_right=(H - 13, W - 13))
depth_err = {}
for name, (r, c) in regions.items():
    patch = Z[r - 4:r + 5, c - 4:c + 5][~lm[r - 4:r + 5, c - 4:c + 5]]
    depth_err[name] = float(np.nanmax(np.abs(patch - 1.0)))
f, cx, cy = env.K[0, 0], env.K[0, 2], env.K[1, 2]
range_at_corner = float(np.sqrt(1 + ((12 - cx) / f) ** 2 + ((12 - cy) / f) ** 2))
report["depth"] = dict(max_abs_error_m=depth_err, euclidean_range_at_corner_would_be=range_at_corner,
                       passed=bool(max(depth_err.values()) < 1e-3))

# %% Six-axis signs: measured landmark motion matches the body-twist prediction
c0, p0 = centroids(T0), project(env.K, T0, fronts)[0]
axes = {}
for i, name in enumerate(("vx", "vy", "vz", "wx", "wy", "wz")):
    xi = np.zeros(6)
    xi[i] = LINEAR_STEP if i < 3 else ANGULAR_STEP
    T = T0 @ se3_exp(xi)
    dm, dp = centroids(T) - c0, project(env.K, T, fronts)[0] - p0
    agree = np.sum(dm * dp, axis=1) > 0
    axes[name] = dict(mean_measured_px=dm.mean(0).tolist(), mean_predicted_px=dp.mean(0).tolist(),
                      max_error_px=float(np.nanmax(np.linalg.norm(dm - dp, axis=1))), sign_agrees=bool(agree.all()))
report["coordinate_basis"] = dict(axes=axes, positive_x_moves_scene_left=bool(axes["vx"]["mean_measured_px"][0] < 0),
                                  passed=bool(all(a["sign_agrees"] and a["max_error_px"] < 1.0 for a in axes.values())
                                              and axes["vx"]["mean_measured_px"][0] < 0))

# %% Render-buffer latency: does one capture after a pose change show the new pose?
centroids(T0)
Tj = T0 @ se3_exp([JUMP, 0, 0, 0, 0, 0])
m = centroids(Tj)
e_new = float(np.nanmax(np.linalg.norm(m - project(env.K, Tj, fronts)[0], axis=1)))
e_old = float(np.nanmax(np.linalg.norm(m - project(env.K, T0, fronts)[0], axis=1)))
report["latency"] = dict(renders_per_capture=cfg.renders_per_capture, error_vs_new_pose_px=e_new,
                         error_vs_previous_pose_px=e_old, stale=bool(e_old < e_new),
                         passed=bool(e_new < 1.0),
                         note="if stale, raise renders_per_capture and rerun; never relabel old buffers")

# %% Write report
report["all_passed"] = all(v["passed"] for v in report.values() if isinstance(v, dict) and "passed" in v)
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT_DIR, "acceptance_plane.json").write_text(json.dumps(report, indent=2, default=str))
print(json.dumps({k: v.get("passed") for k, v in report.items() if isinstance(v, dict) and "passed" in v}, indent=2))
env.close()
app.close()
