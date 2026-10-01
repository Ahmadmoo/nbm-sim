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

# Isaac Lab release line -> Isaac Sim release it was checked against
TESTED = {"2.3": "5.1", "3.0": "6.1"}
_APP = None


def _version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def isaaclab_major():
    v = _version("isaaclab")
    if v is None:
        raise RuntimeError("cannot read the installed Isaac Lab version (importlib.metadata 'isaaclab')")
    return int(v.split(".")[0])


def launch_app(cfg):
    """Start Isaac Sim once per process (GUI when ``cfg.display``). Must run before any Isaac import.

    Isaac Lab 3.x opens a window only when the Kit visualizer is requested; 2.x uses ``headless``.
    """
    global _APP
    if _APP is None:
        from isaaclab.app import AppLauncher
        kw = dict(headless=not cfg.display, enable_cameras=True, device=cfg.device)
        if isaaclab_major() >= 3 and cfg.display:
            kw["visualizer"] = ["kit"]
        _APP = AppLauncher(**kw).app
    return _APP


def isaac_versions():
    return {p: _version(p) for p in ("isaaclab", "isaaclab_physx", "isaacsim", "torch", "numpy", "h5py",
                                     "warp-lang")}


class IsaacCompat:
    """Resolves the version-dependent details explicitly instead of guessing from names.

    | Item             | Isaac Lab 2.3 / Sim 5.1 | Isaac Lab 3.0 / Sim 6.1                  |
    |------------------|-------------------------|------------------------------------------|
    | quaternion order | (w, x, y, z)            | (x, y, z, w)                             |
    | HDR channel      | ``HdrColor`` annotator  | ``rgb_hdr`` (HdrColor AOV, 3 ch float32) |
    | buffers          | torch tensors           | ``ProxyArray`` (use ``.torch``)          |
    | renderer knobs   | ``CameraCfg`` + carb    | ``IsaacRtxRendererCfg``                  |
    """

    def __init__(self, cfg):
        v = isaac_versions()
        self.versions = v
        lab = v["isaaclab"]
        if lab is None:
            raise RuntimeError("cannot determine the Isaac Lab version")
        self.major = int(lab.split(".")[0])
        line = ".".join(lab.split(".")[:2])
        if line not in TESTED and not cfg.allow_untested_isaac:
            raise RuntimeError(f"Isaac Lab {lab} is not a supported line {sorted(TESTED)}. Set "
                               "allow_untested_isaac=True to proceed; the pose readback and acceptance checks "
                               "then decide whether it works.")
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
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=cfg.base_dt, render_interval=1,
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
        cam_kw = dict(prim_path="/World/NBMCamera", update_period=0.0, width=cfg.width, height=cfg.height,
                      data_types=self.data_types, spawn=spawn)
        if self.compat.major >= 3:
            cam_kw["renderer_cfg"] = self._renderer_cfg_v3()
            self.render_settings = self._render_settings_v3()
        else:
            cam_kw["colorize_semantic_segmentation"] = False
            self.render_settings = self._apply_render_settings_v2()
        self.camera = Camera(CameraCfg(**cam_kw))
        self.sim.reset()
        self.render_settings["read_back"] = self._read_render_settings()
        self._set_viewport(spec)
        self.K = self._effective_K()
        self.capture_count = 0
        self.render_seconds = 0.0

    # ---------- renderer settings ----------
    def _renderer_cfg_v3(self):
        from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
        g = dict(antialiasing_mode=self.cfg.antialiasing, carb_settings=dict(self.cfg.render_settings))
        g.update(self.cfg.rtx_global_settings)
        return IsaacRtxRendererCfg(colorize_semantic_segmentation=False, enable_scene_partitioning=False,
                                   depth_clipping_behavior="none",
                                   global_settings=IsaacRtxRendererGlobalSettingsCfg(**g))

    def _render_settings_v3(self):
        applied = dict(path="IsaacRtxRendererCfg.global_settings", antialiasing=self.cfg.antialiasing,
                       carb_settings=dict(self.cfg.render_settings),
                       rtx_global_settings=dict(self.cfg.rtx_global_settings))
        if self.cfg.rtx_deterministic:
            from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_determinism_settings
            apply_isaac_rtx_determinism_settings()
            applied["determinism"] = "apply_isaac_rtx_determinism_settings (RealTimePathTracing, caches off)"
        return applied

    def _apply_render_settings_v2(self):
        applied = dict(path="replicator + carb")
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
        applied["carb_settings"] = dict(self.cfg.render_settings)
        return applied

    def _read_render_settings(self):
        import carb
        s = carb.settings.get_settings()
        keys = list(self.cfg.render_settings) + ["/rtx/rendermode", "/rtx/post/aa/op", "/rtx/post/tonemap/op",
                                                 "/rtx/rtpt/cached/enabled", "/rtx/rtpt/lightcache/cached/enabled",
                                                 "/rtx-transient/dldenoiser/enabled"]
        return {k: s.get(k) for k in keys}

    def set_dt(self, dt):
        """Keep Isaac's physics step equal to the env base tick (called by ``CameraNBMEnv.set_timing``).

        The scene is static and every timestamp comes from the env tick counter, so this keeps the two
        clocks in the same units; Isaac's own ``current_time`` still is not episode time (it advances once
        per capture and per warm-up render).
        """
        if self.compat.major >= 3:
            # 3.x has no public setter; PhysxManager.step() reads sim.cfg.dt on every call
            self.sim.cfg.dt = dt
            prim = self.sim.stage.GetPrimAtPath(self.sim.cfg.physics_prim_path)
            attr = prim.GetAttribute("physxScene:timeStepsPerSecond") if prim.IsValid() else None
            if attr:
                attr.Set(int(round(1.0 / dt)))
            self.sim.set_setting("/persistent/simulation/minFrameRate", int(round(1.0 / dt)))
        else:
            self.sim.set_simulation_dt(physics_dt=dt, rendering_dt=dt * self.sim.cfg.render_interval)
            self.sim.cfg.dt = dt
        got = float(self.sim.get_physics_dt())
        if abs(got - dt) > 1e-12:
            raise RuntimeError(f"Isaac physics dt is {got}, expected {dt}")
        return got

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
        self.camera.update(dt=0.0, force_recompute=True)
        for _ in range(self.cfg.renders_per_capture - 1):
            # each extra render bumps the render generation, so the camera pumps the renderer again
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
            out["depth_t"] = dt.nan_to_num(nan=1e4, posinf=1e4, neginf=1e4)   # warp input only
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
                    isaac_physics_dt=float(self.sim.get_physics_dt()),
                    clock="episode time = env base ticks; Isaac physics dt = base_dt; Isaac current_time unused",
                    depth_semantics="distance_to_image_plane (camera-forward Z); invalid -> NaN",
                    physics="static scene; physics steps do not change scene state")

    def close(self):
        pass
