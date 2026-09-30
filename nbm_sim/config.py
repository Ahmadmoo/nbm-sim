"""Editable experiment configuration (spec §20.1). All values are proposed defaults."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

PROTOCOLS = ("events_only", "events_rgb_known_pose", "events_rgbd_known_pose", "oracle")
EVENT_MODES = ("reference", "accelerated")


@dataclass
class SimConfig:
    scene_id: str = "textured_cube"
    seed: int = 0
    num_envs: int = 1

    width: int = 640
    height: int = 480
    horizontal_fov_deg: float = 60.0
    near_clip: float = 0.05
    far_clip: float = 5.0

    initial_position: tuple | None = None         # None = scene default
    initial_look_at: tuple | None = None          # None = target bounding-box center
    initial_T_wc: np.ndarray | None = None        # saved validated pose; overrides position/look-at

    base_dt: float = 0.001
    event_render_dt: float = 0.001
    action_dt: float = 0.05
    rgb_dt: float = 0.05
    depth_dt: float = 0.05
    keyframe_dt: float = 0.008                    # accelerated mode only

    event_backend: str = "evis"
    event_mode: str = "reference"
    event_threshold: float = 0.20
    event_source: str = "hdr"
    noise_enabled: bool = False
    noise_params: dict = field(default_factory=dict)   # DVSNoiseCfg fields when noise_enabled
    warp_composite: str = "b_primary"
    warp_mv_dilate: int = 0

    max_linear_speed: float = 0.20
    max_angular_speed: float = 0.50
    max_linear_acceleration: float = 0.50
    max_angular_acceleration: float = 1.00
    camera_radius: float = 0.03
    collision_clearance: float = 0.02
    horizontal_workspace_radius: tuple = (0.40, 1.20)
    camera_height: tuple = (0.90, 1.60)

    bootstrap_seconds: float = 1.0
    horizon_seconds: float = 10.0
    max_steps: int = 200

    observation_protocol: str = "events_rgb_known_pose"
    depth_observation_source: str = "none"
    expose_intensity: bool = False

    display: bool = True
    preview_view: str = "overview"                # "overview" or "sensor"
    record: bool = True
    debug_render_dump: bool = False
    output_root: str = "./runs"
    asset_dir: str = "./assets_generated"

    device: str = "cuda:0"
    warmup_renders: int = 30
    renders_per_capture: int = 1                  # raise if the latency check reports stale buffers
    antialiasing: str = "Off"
    render_settings: dict = field(default_factory=lambda: {
        "/rtx/post/motionblur/enabled": False,
        "/rtx/post/dof/enabled": False,
        "/rtx/post/histogram/enabled": False,
    })
    isaac_quat_order: str = "auto"                # "auto", "wxyz" (Isaac Lab 2.x), "xyzw" (3.x)
    allow_untested_isaac: bool = False
    pose_readback_tol: float = 1e-4
    texture_seed: int = 7

    @property
    def K(self) -> np.ndarray:
        fx = (self.width / 2.0) / math.tan(math.radians(self.horizontal_fov_deg) / 2.0)
        return np.array([[fx, 0.0, (self.width - 1) / 2.0],
                         [0.0, fx, (self.height - 1) / 2.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def ticks(self, seconds: float) -> int:
        n = round(seconds / self.base_dt)
        if n < 1 or abs(n * self.base_dt - seconds) > 1e-9 * max(1.0, seconds):
            raise ValueError(f"{seconds} s is not a positive integer multiple of base_dt={self.base_dt}")
        return int(n)

    def validate(self) -> "SimConfig":
        if self.num_envs != 1:
            raise NotImplementedError("first deliverable supports num_envs=1")
        if not (0 < self.width <= 65535 and 0 < self.height <= 65535):
            raise ValueError("uint16 event coordinates require width,height <= 65535")
        for name in ("event_render_dt", "action_dt", "rgb_dt", "depth_dt"):
            self.ticks(getattr(self, name))
        if self.event_mode not in EVENT_MODES:
            raise ValueError(f"event_mode must be one of {EVENT_MODES}")
        if self.event_mode == "accelerated" and self.ticks(self.keyframe_dt) % self.ticks(self.event_render_dt):
            raise ValueError("keyframe_dt must be an integer multiple of event_render_dt")
        if self.event_backend != "evis":
            raise ValueError("only the EVIS backend is integrated")
        if self.event_source not in ("hdr", "ldr"):
            raise ValueError("event_source must be 'hdr' or 'ldr'")
        if self.observation_protocol not in PROTOCOLS:
            raise ValueError(f"observation_protocol must be one of {PROTOCOLS}")
        if self.observation_protocol == "events_rgbd_known_pose" and self.depth_observation_source == "none":
            raise ValueError("events_rgbd_known_pose needs a declared depth_observation_source")
        if self.max_angular_speed * self.max_linear_speed >= self.max_linear_acceleration:
            raise ValueError("max_angular_speed*max_linear_speed must be below max_linear_acceleration, "
                             "otherwise a constant body twist violates the world-frame acceleration limit")
        r0, r1 = self.horizontal_workspace_radius
        z0, z1 = self.camera_height
        if not (0 <= r0 < r1 and z0 < z1):
            raise ValueError("invalid workspace bounds")
        if self.isaac_quat_order not in ("auto", "wxyz", "xyzw"):
            raise ValueError("isaac_quat_order must be auto, wxyz or xyzw")
        if self.renders_per_capture < 1:
            raise ValueError("renders_per_capture must be >= 1")
        return self

    def replace(self, **kw) -> "SimConfig":
        return dataclasses.replace(self, **kw).validate()

    def to_dict(self) -> dict:
        d = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            d[f.name] = v.tolist() if isinstance(v, np.ndarray) else v
        d["K"] = self.K.tolist()
        d["units"] = dict(length="m", time="s", linear_velocity="m/s", angular_velocity="rad/s",
                          angle="rad", depth="m, camera-forward Z")
        return d
