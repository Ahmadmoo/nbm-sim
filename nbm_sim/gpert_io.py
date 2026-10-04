"""Bridge to GPERT (github.com/e3ai/gpert, CVPR 2026) for offline scoring of the stacked stream.

GPERT runs in its own Python environment; this module only writes its input files, runs it as a
subprocess, and reads its exported Gaussians back.

Input format (``EventDataset.load_data_robust_e_nerf`` in GPERT):

| file                     | keys                                                                 |
|--------------------------|----------------------------------------------------------------------|
| raw_events.npz           | position (N,2) [x,y], timestamp (N,) int64 ns, polarity (N,) +-1     |
| camera_poses.npz         | T_wc_position (N,3), T_wc_orientation (N,4) xyzw, T_wc_timestamp ns  |
| camera_calibration.npz   | img_height, img_width, intrinsics (3,3)                              |

Both use the optical camera frame (x right, y down, z forward) and camera-to-world poses, so no axis
change is needed. World coordinates are shifted by ``origin`` (the target center) so GPERT's random
initialization box ``[xyz_min, xyz_max]^3`` covers the target. Stationary intervals are cut from the
time axis (the pose is constant there, so the trajectory stays exact) to avoid empty event windows.
"""
from __future__ import annotations

import json
import pathlib
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from .packet import concat_events


class StreamLog:
    """All events and pose samples so far (the stacked stream), built from packets."""

    def __init__(self):
        self.events, self.pose_t, self.pose_T, self.K, self.size = [], [], [], None, None

    def add(self, packet):
        first = 1 if self.pose_t else 0
        self.events.append(packet["events"])
        self.pose_t.append(np.asarray(packet["pose_t"])[first:])
        self.pose_T.append(np.asarray(packet["pose_T_wc"])[first:])
        self.K, self.size = np.asarray(packet["K"]), tuple(int(v) for v in packet["image_size"])
        return self

    def copy(self):
        c = StreamLog()
        c.events, c.pose_t, c.pose_T, c.K, c.size = list(self.events), list(self.pose_t), list(self.pose_T), \
            self.K, self.size
        return c

    def arrays(self):
        return concat_events(self.events), np.concatenate(self.pose_t), np.concatenate(self.pose_T)


def compress_static(pose_t, pose_T, ev_t, min_gap=0.005, tol=1e-12):
    """Remove stationary intervals (identical consecutive poses lasting >= min_gap) from the time axis.
    Returns new pose times, kept pose indices, new event times, and the removed (start, end) intervals."""
    same = np.abs(np.diff(pose_T.reshape(len(pose_T), -1), axis=0)).max(1) <= tol
    runs, i = [], 0
    while i < len(same):
        if same[i]:
            j = i
            while j < len(same) and same[j]:
                j += 1
            if pose_t[j] - pose_t[i] >= min_gap:
                runs.append((i, j))
            i = j
        else:
            i += 1

    def removed_before(t):
        r = np.zeros_like(t, dtype=np.float64)
        for a, b in runs:
            r += np.clip(t - pose_t[a], 0.0, pose_t[b] - pose_t[a])
        return r

    drop = np.zeros(len(pose_t), bool)
    for a, b in runs:
        drop[a + 1:b + 1] = True
    keep = np.nonzero(~drop)[0]
    return (pose_t[keep] - removed_before(pose_t[keep]), keep, ev_t - removed_before(ev_t),
            [(float(pose_t[a]), float(pose_t[b])) for a, b in runs])


def export_gpert(out_dir, log, origin=(0.0, 0.0, 0.0), compress=True):
    """Write GPERT input files for the whole stacked stream. Returns metadata (also saved as JSON)."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ev, pose_t, pose_T = log.arrays()
    ev_t = np.asarray(ev["t"], np.float64)
    removed = []
    keep = np.arange(len(pose_t))
    if compress:
        pose_t, keep, ev_t, removed = compress_static(pose_t, pose_T, ev_t)
    pose_T = pose_T[keep]
    origin = np.asarray(origin, np.float64)
    ns = lambda t: np.round(np.asarray(t) * 1e9).astype(np.int64)
    np.savez(out / "raw_events.npz", position=np.stack([ev["x"], ev["y"]], 1).astype(np.int64),
             timestamp=ns(ev_t), polarity=np.asarray(ev["p"], np.int8))
    np.savez(out / "camera_poses.npz", T_wc_position=pose_T[:, :3, 3] - origin,
             T_wc_orientation=Rotation.from_matrix(pose_T[:, :3, :3]).as_quat(), T_wc_timestamp=ns(pose_t))
    W, H = log.size
    np.savez(out / "camera_calibration.npz", img_height=H, img_width=W, intrinsics=log.K,
             distortion_model="none", distortion_params=np.zeros(4), bayer_pattern="none")
    meta = dict(n_events=int(len(ev_t)), n_poses=int(len(pose_t)), origin=origin.tolist(),
                removed_static_intervals=removed, duration=float(pose_t[-1] - pose_t[0]),
                note="GPERT's warp model assumes the principal point at W/2 (nbm-sim uses (W-1)/2): 0.5 px")
    (out / "nbm_export.json").write_text(json.dumps(meta, indent=2))
    return meta


def gpert_config(data_root, outdir, half_extent, c=0.20, max_steps=40000, window_s=0.02, n_events=None,
                 duration=None, ckpt=None, seed_gaussians=10000):
    """GPERT config as a dict (based on its cfg/robust_e_nerf/chair.yaml, adapted to nbm-sim)."""
    acc_num = 125000 if not (n_events and duration) else int(max(2000, n_events * window_s / duration))
    return dict(
        data_root=str(data_root), outdir=str(outdir), grut_cfg_path="./cfg/grut_config/configs",
        event_name="raw_events.npz", gsinit_ckpt_path=ckpt or "", train=True, test=False,
        data_type="robust_e-nerf", gsinit_method="checkpoint" if ckpt else "random", test_dir="", test_pose="",
        c=c, max_steps=max_steps, accumulation_num=acc_num, accumulation_time=window_s,
        initial_gaussians=seed_gaussians, use_diff_image_step=min(10000, max_steps // 4), diff_method="once",
        accumulation_method="iwe", dataloader_method="num", interp_method="slerp", log_eps=1e-5,
        log_method="log", background_color="black", xyz_max=half_extent, xyz_min=-half_extent,
        plot_interval=1000, use_focus=True, focus_weight=0.125, depth_grad="posrot", multi_iwe=False,
        ssim_weight=1, use_ssim=True, use_l1=False, use_l2=True, l1_weight=10, l2_weight=500,
        normalize_l1=True, normalize_l2=True, use_masked_l1=False, use_masked_l2=False, l1_mask_weight=0.5,
        l2_mask_weight=0.25, depth_variation_weight=0, variation_weight=0, opacity_reg_weight=0.0,
        scale_reg_weight=0.0, is_color=False, bayer_pattern="RGGB", devide_g=False, bayered_diff=False,
        img_coeff=60, vmin=0.0, vmax=3.0, export_ingp=False, export_ply=True,
        strategy=dict(
            method="GSStrategy", print_stats=True, max_n_gaussians=1500000,
            densify=dict(params="positions", frequency=500, start_iteration=1000, end_iteration=-1,
                         clone_grad_threshold=0.0002, split_grad_threshold=0.0002, relative_size_threshold=0.01,
                         split=dict(n_gaussians=2)),
            prune=dict(frequency=100, start_iteration=100, end_iteration=-1, density_threshold=0.02),
            reset_density=dict(frequency=3000, start_iteration=-1, end_iteration=27000, new_max_density=0.01),
            density_decay=dict(gamma=0.99, start_iteration=500, end_iteration=-1, frequency=50),
            prune_weight=dict(frequency=100, start_iteration=-1, end_iteration=-1, weight_threshold=0.5),
            prune_scale=dict(frequency=100, start_iteration=-1, end_iteration=-1, threshold=1.0)))


def write_config(path, cfg):
    """JSON is valid YAML, so GPERT's yaml.safe_load reads this file."""
    pathlib.Path(path).write_text(json.dumps(cfg, indent=2))
    return str(path)


def load_gpert_points(ply_path, min_opacity=0.5, origin=(0.0, 0.0, 0.0)):
    """Gaussian centers with opacity >= min_opacity (3DGRUT PLY: float32 properties, opacity as logit)."""
    raw = pathlib.Path(ply_path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode().splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("expected a binary little-endian PLY")
    n = next(int(h.split()[2]) for h in header if h.startswith("element vertex"))
    props = [h.split()[2] for h in header if h.startswith("property float")]
    data = np.frombuffer(raw, dtype=np.dtype([(p, "<f4") for p in props]), count=n, offset=end)
    alpha = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float64)))
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64) + np.asarray(origin)
    return xyz[alpha >= min_opacity], alpha[alpha >= min_opacity]


def run_gpert(gpert_root, python_exe, config_path):
    """Run GPERT's ``scripts/run.py`` and return the path of the newest ``export_last.ply``."""
    cfg = json.loads(pathlib.Path(config_path).read_text())
    subprocess.run([python_exe, "scripts/run.py", "--config", str(pathlib.Path(config_path).resolve())],
                   cwd=gpert_root, check=True)
    runs = sorted(pathlib.Path(cfg["outdir"]).glob("*/export_last.ply"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"no export_last.ply under {cfg['outdir']}")
    return str(runs[-1])


def gpert_evaluator(env, workdir, gpert_root, python_exe, tau=0.005, max_steps=40000, min_opacity=0.5,
                    margin=0.05, surface="accessible_surface"):
    """``evaluate(volume, log)`` for ``branch_gains``: export the stacked stream, train GPERT from scratch,
    score its Gaussian centers with nbm-sim's target-only F-score (evaluator-only)."""
    from .evaluation import reconstruction_metrics
    lo, hi = env.spec.task_region(margin)
    origin = (np.asarray(lo) + np.asarray(hi)) / 2
    half = float((np.asarray(hi) - np.asarray(lo)).max() / 2)
    geom, region = env.get_evaluation_state()["geometry"], env.spec.task_region()
    counter = [0]

    def evaluate(volume, log):
        d = pathlib.Path(workdir) / f"eval_{counter[0]:04d}"
        counter[0] += 1
        meta = export_gpert(d / "data", log, origin)
        cfg = gpert_config(d / "data", d / "out", half, max_steps=max_steps, n_events=meta["n_events"],
                           duration=meta["duration"])
        ply = run_gpert(gpert_root, python_exe, write_config(d / "config.yaml", cfg))
        xyz, _ = load_gpert_points(ply, min_opacity, origin)
        return reconstruction_metrics(xyz, geom[surface], region, taus=(tau,))[f"fscore@{tau * 1e3:g}mm"]
    return evaluate
