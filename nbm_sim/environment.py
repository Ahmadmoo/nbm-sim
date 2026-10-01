"""CameraNBMEnv: deterministic reset/step over integer base ticks (spec §6, §7, §11)."""
from __future__ import annotations

import time
import uuid

import numpy as np

from .config import SimConfig
from .geometry import look_at_T_wc
from .motion import MotionController
from .packet import SCHEMA_VERSION, actor_view, concat_events, frozen
from .scene import build_assets, evaluation_geometry, get_scene


class SensorFailure(RuntimeError):
    pass


class CameraNBMEnv:
    """Camera-only environment. ``backend`` defaults to Isaac Sim; tests may inject another
    object with ``capture(T_wc, channels)``, ``K``, ``describe()`` and ``close()``."""

    def __init__(self, config: SimConfig, backend=None):
        self.cfg = config.validate()
        self.spec = get_scene(self.cfg.scene_id)
        if backend is None:
            from .camera import IsaacCameraBackend, launch_app
            launch_app(self.cfg)
            self.assets = build_assets(self.spec, self.cfg.asset_dir, self.cfg.texture_seed)
            backend = IsaacCameraBackend(self.cfg, self.spec, self.assets)
        else:
            self.assets = []
        self.backend = backend
        self.K = np.asarray(backend.K, dtype=np.float64)
        self.motion = MotionController(self.cfg, self.spec.colliders(), self.spec.workspace(self.cfg))
        self.evaluation_geometry = evaluation_geometry(self.spec)
        from .events import EvisEventCamera
        self.events = EvisEventCamera(self.cfg)
        self.recorder = None
        self.episode_id = None
        self._done = True
        self._set_clock()

    # ---------- configuration ----------
    def _set_clock(self):
        c = self.cfg
        self.dt = c.base_dt
        self.n_event = c.ticks(c.event_render_dt)
        self.n_rgb = c.ticks(c.rgb_dt)
        self.n_depth = c.ticks(c.depth_dt)
        self.n_key = c.ticks(c.keyframe_dt) if c.event_mode == "accelerated" else None
        self.horizon_ticks = c.ticks(c.horizon_seconds)

    def set_timing(self, **kw):
        """Change clock/limit settings between episodes (e.g. temporal-convergence tests)."""
        if not self._done:
            raise RuntimeError("change timing only between episodes: call save_episode() or end_episode() first")
        cfg = self.cfg.replace(**kw)
        if cfg.base_dt != self.cfg.base_dt:
            self.backend.set_dt(cfg.base_dt)
        self.cfg = cfg
        self.motion = MotionController(self.cfg, self.spec.colliders(), self.spec.workspace(self.cfg))
        self.events.cfg = self.cfg
        self._set_clock()

    def initial_pose(self):
        c = self.cfg
        if c.initial_T_wc is not None:
            return np.asarray(c.initial_T_wc, dtype=np.float64)
        target = self.spec.target_center if c.initial_look_at is None else c.initial_look_at
        pos = self.spec.default_camera_position if c.initial_position is None else c.initial_position
        return look_at_T_wc(pos, target)

    # ---------- episode ----------
    def reset(self, scene_id=None, seed=None, T_wc=None):
        if scene_id is not None and scene_id != self.cfg.scene_id:
            raise NotImplementedError("one scene per environment instance; create a new env for another scene")
        seed = self.cfg.seed if seed is None else int(seed)
        self.seed = seed
        ss = np.random.SeedSequence(seed)
        streams = dict(zip(("scene", "sensor_noise", "bootstrap", "baseline", "policy", "depth_noise"), ss.spawn(6)))
        self.rng = {k: np.random.default_rng(s) for k, s in streams.items()}
        self._rng_seeds = streams
        if self.recorder is not None:
            self.recorder.close("abandoned")
            self.recorder = None
        self.episode_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{self.cfg.scene_id}_s{seed}_{uuid.uuid4().hex[:6]}"
        T0 = self.initial_pose() if T_wc is None else np.asarray(T_wc, dtype=np.float64)
        self.motion.reset(T0)
        self.tick = 0
        self.policy_steps = 0
        self.step_index = 0
        self.bootstrap_ticks = 0
        self.terminated = self.truncated = False
        self.reason = ""
        self.timing = dict(render=0.0, events=0.0, record=0.0, step_wall=0.0)
        self.counts = dict(events_pos=0, events_neg=0, rgb=0, depth=0, captures=0)
        self._last_key = None
        self._last_key_tick = None
        self._last_eval = None

        for _ in range(self.cfg.warmup_renders):
            self.backend.capture(T0, ())
        self.events.reset(streams["sensor_noise"])
        first = self._capture(T0, 0, events=True, rgb=True, depth=True)
        leftover = self.events.drain()
        if len(leftover["t"]):
            raise SensorFailure(f"event package emitted {len(leftover['t'])} events on its initialization frame")

        if self.cfg.record:
            from .recording import EpisodeRecorder
            self.recorder = EpisodeRecorder(self)
        self._done = False
        pkt = self._packet(t_start=0.0, T_start=T0, pose_t=[0.0], pose_T=[T0], twists=[], rgb=first["rgb_list"],
                           requested=np.zeros(6), limited=False, safety=False, events=concat_events([]),
                           phase="reset", depth_obs=self._depth_observed([first["eval"]]))
        self._emit(pkt, first["eval"])
        return self._public(pkt)

    def run_bootstrap(self, schedule):
        """Execute a saved ``[(velocity, duration), ...]`` schedule before planning (counted in acquisition cost)."""
        return [self.step(v, d, phase="bootstrap") for v, d in schedule]

    def step(self, velocity, duration=None, phase="policy"):
        if self._done:
            raise RuntimeError("episode ended; call reset()")
        w0 = time.perf_counter()
        v = np.asarray(velocity, dtype=np.float64)
        if v.shape != (6,) or not np.all(np.isfinite(v)):
            raise ValueError("velocity must be a finite array of shape (6,)")
        n_req = self.cfg.ticks(self.cfg.action_dt if duration is None else float(duration))
        if self.n_key is not None and (n_req % self.n_key or self.tick % self.n_key):
            raise ValueError(f"accelerated mode: step durations must be multiples of keyframe_dt="
                             f"{self.cfg.keyframe_dt} s so every packet boundary is a keyframe")
        if phase == "bootstrap" and self.policy_steps:
            raise RuntimeError("bootstrap steps must come before the first policy step")
        n = n_req
        if phase == "policy":
            n = min(n_req, self.horizon_ticks - (self.tick - self.bootstrap_ticks))   # stop at the remaining budget
        limited_cmd, was_limited = self.motion.limit(v)
        t_start, T_start = self.tick * self.dt, self.motion.T_wc.copy()
        pose_t, pose_T, twists, rgb, evals = [t_start], [T_start], [], [], []
        safety = False
        for _ in range(n):
            res = self.motion.tick(limited_cmd, self.dt)
            if not res.accepted:
                self.motion.emergency_stop()
                safety, self.terminated, self.reason = True, True, res.reason
                self.stop_time = self.tick * self.dt
                if self.n_key is not None and self.tick != self._last_key_tick:
                    # close the open keyframe gap at the stop pose so no rendered interval is dropped
                    self._capture(self.motion.T_wc, self.tick, events=True, rgb=False, depth=False)
                break
            self.tick += 1
            pose_t.append(self.tick * self.dt)
            pose_T.append(res.T_wc)
            twists.append(res.twist)
            cap = self._capture_scheduled(res.T_wc, self.tick)
            if cap:
                rgb += cap["rgb_list"]
                if cap["eval"] is not None:
                    evals.append(cap["eval"])
        t_ev = time.perf_counter()
        events = self.events.drain()
        self.timing["events"] += time.perf_counter() - t_ev
        if phase == "bootstrap":
            self.bootstrap_ticks = self.tick
        else:
            self.policy_steps += 1
        if not self.terminated and phase == "policy" and (
                self.policy_steps >= self.cfg.max_steps or self.tick - self.bootstrap_ticks >= self.horizon_ticks):
            self.truncated, self.reason = True, "budget_exhausted"
        pkt = self._packet(t_start=t_start, T_start=T_start, pose_t=pose_t, pose_T=pose_T, twists=twists, rgb=rgb,
                           requested=v, limited=was_limited, safety=safety, events=events, phase=phase,
                           duration=n_req * self.dt, budget_clipped=n < n_req, depth_obs=self._depth_observed(evals))
        self.timing["step_wall"] += time.perf_counter() - w0
        self._emit(pkt, evals)
        if self.terminated or self.truncated:
            self._done = True
        return self._public(pkt)

    # ---------- capture scheduling ----------
    def _capture_scheduled(self, T, tick):
        c = self.cfg
        if c.event_mode == "reference":
            ev = tick % self.n_event == 0
        else:
            ev = tick % self.n_key == 0
        rgb, dep = tick % self.n_rgb == 0, tick % self.n_depth == 0
        if not (ev or rgb or dep):
            return None
        return self._capture(T, tick, events=ev, rgb=rgb, depth=dep)

    def _capture(self, T, tick, events, rgb, depth):
        ch = set()
        accel = self.cfg.event_mode == "accelerated"
        if events:
            ch |= {"hdr"} | ({"mv", "depth_t"} if accel else set())
        if events and accel and tick % self.n_key and not self.terminated:
            raise RuntimeError(f"internal: keyframe capture at non-keyframe tick {tick}")
        if rgb:
            ch.add("rgb")
        if depth:
            ch |= {"depth", "mask"}
        t0 = time.perf_counter()
        out = self.backend.capture(T, tuple(sorted(ch)))
        self.timing["render"] += time.perf_counter() - t0
        self.counts["captures"] += 1
        t = tick * self.dt
        if events:
            f = out["hdr_torch"]
            bad = int((~f.isfinite()).sum()) + int((f < 0).sum())
            if bad:
                raise SensorFailure(f"{bad} non-finite or negative intensity values at t={t:.6f}")
            t1 = time.perf_counter()
            if not accel or self._last_key is None:
                self.events.process(f, t)
            else:
                gap = tick - self._last_key_tick
                k = max(1, round(gap / self.n_event))
                self.events.warp_gap(self._last_key, dict(hdr=f, mv=out["mv"], depth_t=out["depth_t"]),
                                     self._last_key_tick * self.dt, t, k)
            if accel:
                self._last_key = dict(hdr=f, mv=out["mv"], depth_t=out["depth_t"])
                self._last_key_tick = tick
            self.timing["events"] += time.perf_counter() - t1
            self._last_intensity = f
        rgb_list = [(t, T.copy(), out["rgb"])] if rgb else []
        self.counts["rgb"] += len(rgb_list)
        ev = None
        if depth:
            ev = dict(t=t, T_wc=T.copy(), depth=out["depth"], depth_valid=out["depth_valid"], mask=out["mask"])
            self.counts["depth"] += 1
            self._last_eval = ev
        return dict(rgb_list=rgb_list, eval=ev)

    # ---------- packets ----------
    def _depth_observed(self, evals):
        """Co-located simulated depth sensor (``depth_observation_source='simulator'``) at ``depth_dt``:
        GT depth plus declared noise ``k*Z^2`` and dropout. Separate from evaluator ``depth_gt``."""
        if self.cfg.depth_observation_source == "none":
            return None
        rng, k, drop = self.rng["depth_noise"], self.cfg.depth_noise_std_at_1m, self.cfg.depth_dropout
        out = []
        for ev in evals:
            if ev is None:
                continue
            z, valid = ev["depth"].astype(np.float32).copy(), ev["depth_valid"].copy()
            if k > 0:
                z[valid] += (rng.standard_normal(int(valid.sum())) * k * z[valid] ** 2).astype(np.float32)
                valid &= z > 0
            if drop > 0:
                valid &= rng.random(z.shape) >= drop
            z[~valid] = np.nan
            out.append((ev["t"], ev["T_wc"].copy(), z))
        return out

    def _packet(self, t_start, T_start, pose_t, pose_T, twists, rgb, requested, limited, safety, events, phase,
                duration=0.0, budget_clipped=False, depth_obs=None):
        H, W = self.cfg.height, self.cfg.width
        npos = int((events["p"] > 0).sum())
        self.counts["events_pos"] += npos
        self.counts["events_neg"] += len(events["p"]) - npos
        pkt = dict(
            schema_version=SCHEMA_VERSION, episode_id=self.episode_id, step_index=self.step_index, phase=phase,
            t_start=float(t_start), t_end=float(pose_t[-1]), events=events,
            rgb=frozen(np.stack([r[2] for r in rgb]) if rgb else np.zeros((0, H, W, 3), np.uint8)),
            rgb_t=frozen(np.array([r[0] for r in rgb], np.float64)),
            rgb_T_wc=frozen(np.stack([r[1] for r in rgb]) if rgb else np.zeros((0, 4, 4))),
            K=frozen(self.K), image_size=frozen(np.array([W, H], np.int64)),
            T_wc_start=frozen(T_start), T_wc_end=frozen(pose_T[-1]),
            pose_t=frozen(np.array(pose_t, np.float64)), pose_T_wc=frozen(np.stack(pose_T)),
            executed_twist=frozen(np.array(twists, np.float64).reshape(-1, 6)),
            requested_velocity=frozen(np.asarray(requested, np.float64)),
            command_limited=bool(limited), safety_intervention=bool(safety),
            terminated=bool(self.terminated), truncated=bool(self.truncated), reason=self.reason,
            diagnostics=dict(n_events=len(events["t"]), n_pos=npos, requested_duration=float(duration),
                             budget_clipped=bool(budget_clipped),
                             executed_duration=float(pose_t[-1] - t_start), path_length=self.motion.path_length,
                             rotation_travel=self.motion.rotation_travel, acquisition_time=self.tick * self.dt,
                             policy_steps=self.policy_steps))
        if depth_obs is not None:
            pkt["depth_observed"] = frozen(np.stack([d[2] for d in depth_obs]) if depth_obs
                                           else np.zeros((0, H, W), np.float32))
            pkt["depth_observed_t"] = frozen(np.array([d[0] for d in depth_obs], np.float64))
            pkt["depth_observed_T_wc"] = frozen(np.stack([d[1] for d in depth_obs]) if depth_obs
                                                else np.zeros((0, 4, 4)))
        if self.cfg.expose_intensity and getattr(self, "_last_intensity", None) is not None:
            f = self._last_intensity[0].detach().cpu().numpy()
            pkt["intensity"] = frozen((f[..., :3] @ np.array([0.2126, 0.7152, 0.0722])).astype(np.float32))
        self.step_index += 1
        return pkt

    def _emit(self, pkt, evals):
        if self.recorder is not None:
            t0 = time.perf_counter()
            self.recorder.append(pkt, evals if isinstance(evals, list) else [evals])
            self.timing["record"] += time.perf_counter() - t0

    def _public(self, pkt):
        return actor_view(pkt, self.cfg.observation_protocol, self.cfg.expose_intensity)

    # ---------- state access ----------
    def camera_state(self):
        """Known-pose camera state available to planners (``pose_gt`` under the known-pose assumption)."""
        return dict(T_wc=self.motion.T_wc.copy(), t=self.tick * self.dt, v_world=self.motion.v_w.copy(),
                    w_world=self.motion.w_w.copy(), path_length=self.motion.path_length,
                    rotation_travel=self.motion.rotation_travel, target_center_prior=self.spec.target_center.copy())

    def get_evaluation_state(self):
        """Evaluator-only data: latest GT depth/mask and the target reference geometry."""
        return dict(latest=self._last_eval, geometry=self.evaluation_geometry,
                    task_region=self.spec.task_region())

    def get_state(self):
        return dict(tick=self.tick, policy_steps=self.policy_steps, step_index=self.step_index,
                    bootstrap_ticks=self.bootstrap_ticks, terminated=self.terminated, truncated=self.truncated,
                    reason=self.reason, motion=self.motion.get_state(), events=self.events.get_state(),
                    last_key_tick=self._last_key_tick,
                    last_key=None if self._last_key is None else {k: v.clone() for k, v in self._last_key.items()},
                    rng={k: r.bit_generator.state for k, r in self.rng.items()},
                    limitations="renderer history is not captured; restore re-renders from the restored pose")

    def set_state(self, s):
        self.tick, self.policy_steps, self.step_index = s["tick"], s["policy_steps"], s["step_index"]
        self.bootstrap_ticks = s["bootstrap_ticks"]
        self.terminated, self.truncated, self.reason = s["terminated"], s["truncated"], s["reason"]
        self.motion.set_state(s["motion"])
        self.events.set_state(s["events"])
        self._last_key_tick = s["last_key_tick"]
        self._last_key = None if s["last_key"] is None else {k: v.clone() for k, v in s["last_key"].items()}
        for k, st in s["rng"].items():
            self.rng[k].bit_generator.state = st
        self._done = self.terminated or self.truncated

    def describe(self):
        return dict(backend=self.backend.describe(), event_package=self.events.describe(),
                    assets=self.assets, scene=self.spec.scene_id)

    def save_episode(self, output_dir=None):
        """Write the episode and end it. Further ``step`` calls need a new ``reset``."""
        if self.recorder is None:
            raise RuntimeError("recording is disabled (record=False) or no episode is active")
        path = self.recorder.close("complete", output_dir)
        self.recorder = None
        self._done = True
        return path

    def end_episode(self):
        """End the episode without saving (recording, if any, is closed with status 'ended_unsaved')."""
        if self.recorder is not None:
            self.recorder.close("ended_unsaved")
            self.recorder = None
        self._done = True

    def close(self):
        if self.recorder is not None:
            self.recorder.close("closed_without_save")
            self.recorder = None
        self.backend.close()
