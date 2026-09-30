"""Episode recording, manifest, and replay (spec §14)."""
from __future__ import annotations

import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np

_STR = h5py.string_dtype()


def _append(group, name, arr, chunk_rows=None, compression="gzip"):
    arr = np.asarray(arr)
    if name not in group:
        rows = chunk_rows or max(1, min(65536, int(4e6 // max(1, arr[0:1].nbytes or 1))))
        group.create_dataset(name, shape=(0,) + arr.shape[1:], maxshape=(None,) + arr.shape[1:], dtype=arr.dtype,
                             chunks=(rows,) + arr.shape[1:], compression=compression)
    ds = group[name]
    n = ds.shape[0]
    if len(arr):
        ds.resize(n + len(arr), axis=0)
        ds[n:] = arr
    return n, n + len(arr)


def _gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or "unavailable"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def build_manifest(env, status):
    import importlib.metadata as md
    vers = {}
    for p in ("isaacsim", "isaaclab", "dvs_gen", "torch", "numpy", "h5py", "scipy", "warp-lang"):
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    cuda = None
    try:
        import torch
        cuda = dict(runtime=torch.version.cuda, device=torch.cuda.get_device_name(0) if torch.cuda.is_available()
                    else None, peak_vram_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available()
                    else None)
    except Exception:
        pass
    return dict(status=status, episode_id=env.episode_id, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                os=platform.platform(), python=sys.version, gpu=_gpu_info(), cuda=cuda, packages=vers,
                description=env.describe(), seeds=dict(episode=env.seed,
                                                       streams={k: list(map(int, s.spawn_key)) for k, s in
                                                                env._rng_seeds.items()}),
                milestone_validated=None)


class EpisodeRecorder:
    """Streams packets into ``episode.h5`` with separate observation and evaluation groups."""

    def __init__(self, env):
        self.env = env
        root = pathlib.Path(env.cfg.output_root)
        self.run_dir = root / env.episode_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "config.json").write_text(json.dumps(env.cfg.to_dict(), indent=2, default=str))
        self._write_manifest("running")
        self.f = h5py.File(self.run_dir / "episode.h5", "w")
        self.f.attrs.update(schema_version="1.0", episode_id=env.episode_id, scene_id=env.cfg.scene_id,
                            seed=env.seed, width=env.cfg.width, height=env.cfg.height,
                            observation_protocol=env.cfg.observation_protocol, pose_convention="T_wc optical frame",
                            time_units="s", length_units="m")
        self.obs, self.steps = self.f.create_group("obs"), self.f.create_group("steps")
        self.traj, self.ev = self.f.create_group("traj"), self.f.create_group("eval")
        self.f["obs/K"] = env.K
        g = self.f.create_group("eval/geometry")
        for k, v in env.evaluation_geometry.items():
            if isinstance(v, np.ndarray):
                g.create_dataset(k, data=v, compression="gzip")
            else:
                g.attrs[k] = v
        self.f.flush()

    def _write_manifest(self, status):
        (self.run_dir / "manifest.json").write_text(json.dumps(build_manifest(self.env, status), indent=2,
                                                               default=str))

    def append(self, pkt, evals):
        e = pkt["events"]
        es = [_append(self.obs, f"events/{k}", e[k], chunk_rows=1 << 16) for k in ("x", "y", "t", "p")][0]
        H, W = self.env.cfg.height, self.env.cfg.width
        rs = _append(self.obs, "rgb", pkt["rgb"].reshape(-1, H, W, 3), chunk_rows=1)
        _append(self.obs, "rgb_t", pkt["rgb_t"], chunk_rows=1024)
        _append(self.obs, "rgb_T_wc", pkt["rgb_T_wc"].reshape(-1, 4, 4), chunk_rows=1024)
        first = 0 if pkt["phase"] == "reset" else 1
        ps = _append(self.traj, "pose_t", pkt["pose_t"][first:], chunk_rows=4096)
        _append(self.traj, "pose_T_wc", pkt["pose_T_wc"][first:], chunk_rows=4096)
        _append(self.traj, "executed_twist", pkt["executed_twist"], chunk_rows=4096)
        s = self.steps
        row = dict(step_index=[pkt["step_index"]], t_start=[pkt["t_start"]], t_end=[pkt["t_end"]],
                   requested_velocity=pkt["requested_velocity"][None],
                   requested_duration=[pkt["diagnostics"]["requested_duration"]],
                   command_limited=[pkt["command_limited"]], safety_intervention=[pkt["safety_intervention"]],
                   terminated=[pkt["terminated"]], truncated=[pkt["truncated"]],
                   event_range=[es], rgb_range=[rs], pose_range=[(ps[0] - first, ps[1])])
        for k, v in row.items():
            _append(s, k, np.asarray(v), chunk_rows=1024, compression=None)
        _append(s, "phase", np.array([pkt["phase"]], dtype=object).astype(_STR), chunk_rows=1024, compression=None)
        _append(s, "reason", np.array([pkt["reason"]], dtype=object).astype(_STR), chunk_rows=1024, compression=None)
        for ev in evals:
            if ev is None:
                continue
            _append(self.ev, "depth", ev["depth"][None], chunk_rows=1)
            _append(self.ev, "depth_valid", ev["depth_valid"][None], chunk_rows=1)
            _append(self.ev, "target_mask", ev["mask"][None], chunk_rows=1)
            _append(self.ev, "depth_t", [ev["t"]], chunk_rows=1024)
            _append(self.ev, "depth_T_wc", ev["T_wc"][None], chunk_rows=1024)
        self.f.flush()

    def close(self, status, output_dir=None):
        env = self.env
        self.f.attrs["status"] = status
        self.f.close()
        metrics = dict(evaluation_protocol="events + known poses; target-only metrics require a reconstruction",
                       counts=env.counts, timing_wall_s=env.timing, acquisition_time_s=env.tick * env.dt,
                       path_length_m=env.motion.path_length, rotation_travel_rad=env.motion.rotation_travel,
                       terminated=env.terminated, truncated=env.truncated, reason=env.reason,
                       captures=env.counts["captures"])
        (self.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        self._write_manifest(status)
        if output_dir is None:
            return str(self.run_dir)
        dst = pathlib.Path(output_dir) / self.run_dir.name
        if dst.exists():
            raise FileExistsError(f"{dst} exists; refusing to overwrite")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.run_dir), str(dst))
        return str(dst)


# ---------- reading and replay ----------

def load_episode(run_dir):
    """Load a recorded episode into numpy arrays (no rerendering)."""
    p = pathlib.Path(run_dir)
    with h5py.File(p / "episode.h5", "r") as f:
        def grab(g):
            out = {}
            g.visititems(lambda k, v: out.__setitem__(k, v[()]) if isinstance(v, h5py.Dataset) else None)
            return out
        data = dict(attrs=dict(f.attrs), obs=grab(f["obs"]), steps=grab(f["steps"]), traj=grab(f["traj"]),
                    eval=grab(f["eval"]))
    data["config"] = json.loads((p / "config.json").read_text())
    return data


def iter_packets(data):
    """Rebuild per-step observation packets from a loaded episode."""
    o, s, tr = data["obs"], data["steps"], data["traj"]
    ev_keys = ("x", "y", "t", "p")
    for i in range(len(s["step_index"])):
        e0, e1 = s["event_range"][i]
        r0, r1 = s["rgb_range"][i]
        p0, p1 = s["pose_range"][i]
        yield dict(step_index=int(s["step_index"][i]), phase=s["phase"][i].decode() if isinstance(s["phase"][i], bytes)
                   else s["phase"][i], t_start=float(s["t_start"][i]), t_end=float(s["t_end"][i]),
                   events={k: o.get(f"events/{k}", np.zeros(0))[e0:e1] for k in ev_keys},
                   rgb=o["rgb"][r0:r1] if "rgb" in o else None, rgb_t=o.get("rgb_t", np.zeros(0))[r0:r1],
                   pose_t=tr["pose_t"][p0:p1], pose_T_wc=tr["pose_T_wc"][p0:p1],
                   requested_velocity=s["requested_velocity"][i], requested_duration=float(s["requested_duration"][i]))


def command_schedule(data):
    """``[(phase, velocity, duration), ...]`` for every non-reset step."""
    return [(p["phase"], p["requested_velocity"], p["requested_duration"]) for p in iter_packets(data)
            if p["phase"] != "reset"]


def rerun(env, run_dir):
    """Re-execute a recorded command schedule from its reset state; returns the new packets."""
    data = load_episode(run_dir)
    pkts = [env.reset(seed=int(data["attrs"]["seed"]))]
    for phase, v, d in command_schedule(data):
        pkts.append(env.step(v, d, phase=phase))
    return pkts


def compare_event_streams(a, b, t_tol=0.0):
    """Exact comparison of two event dicts; returns a summary dict."""
    same_n = len(a["t"]) == len(b["t"])
    eq = same_n and all(np.array_equal(a[k], b[k]) for k in ("x", "y", "p")) and \
        np.all(np.abs(np.asarray(a["t"]) - np.asarray(b["t"])) <= t_tol)
    return dict(n_a=len(a["t"]), n_b=len(b["t"]), identical=bool(eq))


def compare_episodes(run_a, run_b, pose_tol=1e-9):
    """Compare two recorded runs (e.g. GUI vs headless, or replay vs original)."""
    A, B = load_episode(run_a), load_episode(run_b)
    ev = compare_event_streams({k: A["obs"].get(f"events/{k}", []) for k in "xytp"},
                               {k: B["obs"].get(f"events/{k}", []) for k in "xytp"})
    ta, tb = A["traj"]["pose_T_wc"], B["traj"]["pose_T_wc"]
    dpose = float(np.max(np.abs(ta - tb))) if ta.shape == tb.shape else float("inf")
    return dict(events=ev, max_pose_abs_diff=dpose, poses_match=dpose <= pose_tol)


def write_preview(run_dir, out_path=None, fps=20, window=0.05):
    """Optional side-by-side RGB | event preview video (lossy; never a measurement source)."""
    import cv2
    data = load_episode(run_dir)
    W, H = int(data["attrs"]["width"]), int(data["attrs"]["height"])
    out_path = out_path or os.path.join(run_dir, "preview.mp4")
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * W, H))
    o = data["obs"]
    t = o.get("events/t", np.zeros(0))
    for i, tc in enumerate(o.get("rgb_t", [])):
        m = (t > tc - window) & (t <= tc)
        ev = np.full((H, W, 3), 255, np.uint8)
        pos = o["events/p"][m] > 0
        ev[o["events/y"][m][pos], o["events/x"][m][pos]] = (0, 0, 255)
        ev[o["events/y"][m][~pos], o["events/x"][m][~pos]] = (255, 0, 0)
        vw.write(np.concatenate([cv2.cvtColor(o["rgb"][i], cv2.COLOR_RGB2BGR), ev], axis=1))
    vw.release()
    return out_path
