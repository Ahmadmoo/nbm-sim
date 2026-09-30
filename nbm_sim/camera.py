"""Isaac Sim / Isaac Lab adapter: app launch, scene build, sensor configuration and capture (spec §3, §5, §9, §10).

Only this module (and ``scene.spawn_scene``) touches Isaac APIs. Everything is converted to
the public optical-frame convention at this boundary.
"""
from __future__ import annotations

import importlib.metadata as md
import time

import numpy as np

from .geometry import (K_isaac_to_public, K_public_to_isaac, OPTICAL_TO_GL, optical_to_gl, quat_from_order,
                       quat_to_order, quat_wxyz_to_rotmat, rotation_angle_between, rotmat_to_quat_wxyz)

TESTED = {"isaaclab": "2.3", "isaacsim": "5.1"}
_APP = None


def launch_app(cfg):
    """Start Isaac Sim once per process (GUI when ``cfg.display``). Must run before any Isaac import."""
    global _APP
    if _APP is None:
        from isaaclab.app import AppLauncher
        _APP = AppLauncher(headless=not cfg.display, enable_cameras=True, device=cfg.device).app
    return _APP


def _version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def isaac_versions():
    v = {p: _version(p) for p in ("isaaclab", "isaacsim", "torch", "numpy", "h5py", "warp-lang")}
    if v["isaaclab"] is None:
        try:
            import isaaclab
            v["isaaclab"] = getattr(isaaclab, "__version__", None)
        except ImportError:
            pass
    return v


class IsaacCompat:
    """Resolves the version-dependent details explicitly instead of guessing from names."""

    def __init__(self, cfg):
        v = isaac_versions()
        self.versions = v
        lab = v["isaaclab"]
        if lab is None:
            raise RuntimeError("cannot determine the Isaac Lab version; set isaac_quat_order explicitly "
                               "and allow_untested_isaac=True if you know it")
        self.major = int(str(lab).split(".")[0])
        if not str(lab).startswith(TESTED["isaaclab"]) and not cfg.allow_untested_isaac:
            raise RuntimeError(f"Isaac Lab {lab} is not the tested pair (Isaac Lab {TESTED['isaaclab']}.x / Isaac "
                               f"Sim {TESTED['isaacsim']}, which EVIS targets). Set allow_untested_isaac=True to "
                               "proceed; the pose readback and acceptance checks then decide whether it works.")
        auto = "wxyz" if self.major < 3 else "xyzw"
        self.quat_order = auto if cfg.isaac_quat_order == "auto" else cfg.isaac_quat_order
        self.hdr_channel = "HdrColor" if self.major < 3 else "rgb_hdr"

    @staticmethod
    def tensor(x):
        return x.torch if hasattr(x, "torch") else x


class IsaacCameraBackend:
    """One pinhole camera in a static scene. ``capture`` sets the optical-frame pose, renders, and
    returns the requested channels with the pose read back from the sensor."""

    def __init__(self, cfg, spec, asset_table):
        launch_app(cfg)
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg

        from .scene import spawn_scene

        self.cfg = cfg
        self.compat = IsaacCompat(cfg)
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=cfg.event_render_dt, render_interval=1,
                                                                        device=cfg.device))
        spawn_scene(spec, asset_table, self.compat.quat_order)
        types = ["rgb", "distance_to_image_plane", "semantic_segmentation"]
        types.append(self.compat.hdr_channel if cfg.event_source == "hdr" else "rgb")
        if cfg.event_mode == "accelerated":
            types.append("motion_vectors")
        self.data_types = list(dict.fromkeys(types))
        spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=K_public_to_isaac(cfg.K).ravel().tolist(), width=cfg.width, height=cfg.height,
            clipping_range=(cfg.near_clip, cfg.far_clip))
        self.camera = Camera(CameraCfg(prim_path="/World/NBMCamera", update_period=0.0, width=cfg.width,
                                       height=cfg.height, data_types=self.data_types,
                                       colorize_semantic_segmentation=False, spawn=spawn))
        self.render_settings = self._apply_render_settings()
        self.sim.reset()
        self._set_viewport(spec)
        self.K = self._effective_K()
        self.capture_count = 0
        self.render_seconds = 0.0

    def _apply_render_settings(self):
        applied = {}
        try:
            import omni.replicator.core as rep
            rep.settings.set_render_rtx_realtime(antialiasing=self.cfg.antialiasing)
            applied["antialiasing"] = self.cfg.antialiasing
        except Exception as ex:
            applied["antialiasing"] = f"NOT APPLIED: {ex}"
        import carb
        s = carb.settings.get_settings()
        for k, v in self.cfg.render_settings.items():
            s.set(k, v)
            applied[k] = {"requested": v, "read_back": s.get(k)}
        for k in ("/rtx/rendermode", "/rtx/post/aa/op", "/rtx/post/tonemap/op"):
            applied[k] = {"read_back": s.get(k)}
        return applied

    def _set_viewport(self, spec):
        if not self.cfg.display:
            return
        if self.cfg.preview_view == "sensor":
            from omni.kit.viewport.utility import get_active_viewport
            get_active_viewport().camera_path = "/World/NBMCamera"
        else:
            c = spec.target_center
            self.sim.set_camera_view(eye=(c[0] + 2.0, c[1] - 2.0, c[2] + 1.2), target=tuple(c))

    def _effective_K(self):
        Ki = self.compat.tensor(self.camera.data.intrinsic_matrices)[0].detach().cpu().numpy()
        K = K_isaac_to_public(Ki)
        if np.max(np.abs(K - self.cfg.K)) > 1e-3:
            raise RuntimeError(f"effective K differs from configured K:\n{K}\nvs\n{self.cfg.K}")
        return K

    def _set_pose(self, T_wc):
        import torch
        T_gl = optical_to_gl(T_wc)
        q = quat_to_order(rotmat_to_quat_wxyz(T_gl[:3, :3]), self.compat.quat_order)
        dev = self.cfg.device
        self.camera.set_world_poses(torch.tensor(T_gl[None, :3, 3], dtype=torch.float32, device=dev),
                                    torch.tensor(q[None], dtype=torch.float32, device=dev), convention="opengl")

    def _read_pose(self):
        d = self.camera.data
        p = self.compat.tensor(d.pos_w)[0].detach().cpu().numpy().astype(np.float64)
        q = quat_from_order(self.compat.tensor(d.quat_w_ros)[0].detach().cpu().numpy().astype(np.float64),
                            self.compat.quat_order)
        T = np.eye(4)
        T[:3, :3] = quat_wxyz_to_rotmat(q)
        T[:3, 3] = p
        return T

    def capture(self, T_wc, channels):
        """Render at ``T_wc`` and return ``{channel: array}``. Channels: hdr, rgb, depth, mask, mv."""
        t0 = time.perf_counter()
        self._set_pose(T_wc)
        self.sim.step(render=True)
        for _ in range(self.cfg.renders_per_capture - 1):
            self.sim.render()
        self.camera.update(dt=0.0, force_recompute=True)
        self.render_seconds += time.perf_counter() - t0
        self.capture_count += 1
        T_read = self._read_pose()
        dp = np.linalg.norm(T_read[:3, 3] - T_wc[:3, 3])
        dr = rotation_angle_between(T_read[:3, :3], T_wc[:3, :3])
        if dp > self.cfg.pose_readback_tol or dr > self.cfg.pose_readback_tol:
            raise RuntimeError(f"sensor pose readback mismatch ({dp:.2e} m, {dr:.2e} rad): check quaternion order "
                               f"({self.compat.quat_order}) and frame conversion")
        o = self.camera.data.output
        out = {"T_wc_readback": T_read}
        if "hdr" in channels or "hdr_torch" in channels:
            src = self.compat.hdr_channel if self.cfg.event_source == "hdr" else "rgb"
            f = self.compat.tensor(o[src])[..., :3].float().clone()
            out["hdr_torch"] = f
        if "rgb" in channels:
            out["rgb"] = self.compat.tensor(o["rgb"])[0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        if "depth" in channels or "depth_t" in channels:
            dt = self.compat.tensor(o["distance_to_image_plane"]).float()
            dt = dt[..., 0] if dt.dim() == 4 else dt
            out["depth_t"] = dt.clone()
            z = dt[0].detach().cpu().numpy().astype(np.float32)
            valid = np.isfinite(z) & (z > 0) & (z >= self.cfg.near_clip) & (z <= self.cfg.far_clip)
            out["depth"] = np.where(valid, z, np.nan).astype(np.float32)
            out["depth_valid"] = valid
        if "mask" in channels or "semantic" in channels:
            sem = self._semantic_masks(o)
            out["mask"] = sem.get("target", np.zeros((self.cfg.height, self.cfg.width), bool))
            if "semantic" in channels:
                out["semantic"] = sem
        if "mv" in channels:
            out["mv"] = self.compat.tensor(o["motion_vectors"])[..., :2].float().nan_to_num().clone()
        return out

    def _semantic_masks(self, o):
        """``{class label: bool mask}`` from the semantic segmentation annotator."""
        seg = self.compat.tensor(o["semantic_segmentation"])[0].detach().cpu().numpy()
        seg = seg[..., 0] if seg.ndim == 3 else seg
        info = self.camera.data.info
        info = info[0] if isinstance(info, (list, tuple)) else info
        mapping = (info.get("semantic_segmentation") or {}).get("idToLabels", {})
        masks = {}
        for k, v in mapping.items():
            label = v.get("class") if isinstance(v, dict) else v
            if label is not None:
                masks[label] = masks.get(label, np.zeros(seg.shape, bool)) | (seg == int(k))
        return masks

    def describe(self):
        return dict(versions=self.compat.versions, quat_order=self.compat.quat_order,
                    hdr_channel=self.compat.hdr_channel, data_types=self.data_types,
                    render_settings=self.render_settings, renders_per_capture=self.cfg.renders_per_capture,
                    K_effective=self.K.tolist(), optical_to_gl=OPTICAL_TO_GL.tolist(),
                    depth_semantics="distance_to_image_plane (camera-forward Z); invalid -> NaN",
                    physics="static scene; physics steps do not change scene state")

    def close(self):
        pass
