"""Thin wrapper around the EVIS event-camera package (spec §8).

EVIS (``dvs_gen``) generates every event. This module only configures it, feeds it the
captured intensity frames, collects what it emits, and converts the output format.
No event-generation logic lives here.
"""
from __future__ import annotations

import copy
import dataclasses
import inspect

import numpy as np

from .packet import concat_events


class _PacketSink:
    """Duck-typed stand-in for ``GeneralDVSRecorder``: keeps the package's emitted events in memory."""

    def __init__(self):
        self.chunks = []

    def record(self, camera_name, env_ids, xs, ys, ps, t):
        if xs.numel() == 0:
            return
        x = xs.cpu().numpy().astype(np.uint16)
        self.chunks.append(dict(x=x, y=ys.cpu().numpy().astype(np.uint16),
                                t=np.full(x.shape, t, np.float64), p=ps.cpu().numpy().astype(np.int8)))

    def drain(self):
        ev = concat_events(self.chunks)
        self.chunks = []
        return ev


def package_info():
    """Version/commit of the installed EVIS package and the event-model constants found in its source."""
    import importlib.metadata as md
    import pathlib
    import subprocess

    import dvs_gen
    from dvs_gen.dvs import processor

    info = {"package": "dvs_gen (EVIS)"}
    try:
        info["version"] = md.version("dvs_gen")
    except md.PackageNotFoundError:
        info["version"] = "unknown"
    root = pathlib.Path(dvs_gen.__file__).resolve().parents[1]
    try:
        info["commit"] = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                        text=True, check=True).stdout.strip()
        info["dirty"] = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                            capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        info["commit"] = "unknown"
    src = inspect.getsource(processor.BatchedMultiCamProcessor.__call__)
    info["model_as_found_in_source"] = {
        "luma": "0.2126 R + 0.7152 G + 0.0722 B" if "0.2126" in src and "0.7152" in src and "0.0722" in src
        else "UNVERIFIED: source changed",
        "log_floor": "log(I + 1e-5)" if "1e-5" in src else "UNVERIFIED: source changed",
        "reference_update": "reference latched to current log(I) where |diff| >= threshold"
        if "ref_log_intensity[pos_mask] = log_intensity[pos_mask]" in src else "UNVERIFIED: source changed",
        "events_per_pixel_per_sample": "at most one",
        "timestamp": "sample time of the frame passed to the processor",
        "first_frame": "initializes reference, emits no events",
    }
    return info


class EvisEventCamera:
    """One EVIS processor (camera ``nbm_cam``, env 0) whose sensor state persists across actions."""

    def __init__(self, cfg, seed_seq=None):
        from dvs_gen.dvs import BatchedMultiCamProcessor, DVSNoiseCfg, DVSNoiseModel
        self._Proc, self._NoiseCfg, self._Noise = BatchedMultiCamProcessor, DVSNoiseCfg, DVSNoiseModel
        self.cfg = cfg
        self.sink = _PacketSink()
        self.proc = None
        self.initialized = False
        self.reset(seed_seq)

    def reset(self, seed_seq=None):
        """New episode: a fresh processor; the next frame initializes its reference."""
        noise = None
        if self.cfg.noise_enabled:
            seed = int(seed_seq.generate_state(1)[0]) if seed_seq is not None else 0
            ncfg = self._NoiseCfg(**{**self.cfg.noise_params, "seed": seed})
            noise = self._Noise(ncfg, self.cfg.event_threshold)
        self.proc = self._Proc(self.sink, "nbm_cam", self.cfg.event_threshold, noise=noise)
        self.sink.chunks = []
        self.initialized = False

    def process(self, frame, t):
        """Feed one intensity frame ``(1,H,W,C)`` (torch, float) captured at simulated time ``t``."""
        self.proc(frame, float(t))
        self.initialized = True

    def warp_gap(self, prev, cur, t0, dt_fine, k):
        """Accelerated mode: EVIS motion-vector warp of keyframe gap ``prev -> cur``.

        Feeds the K-1 synthesized frames at ``t0 + i*dt_fine`` and then the real keyframe ``cur``
        at ``t0 + k*dt_fine``, so every event of the gap lands inside ``(t0, t1]``.
        """
        from dvs_gen.warp import bidir_warp_gap
        mids = bidir_warp_gap(prev["hdr"], cur["hdr"], prev["mv"], cur["mv"], k, self.cfg.warp_composite,
                              depthA=prev["depth_t"], depthB=cur["depth_t"], mv_dilate=self.cfg.warp_mv_dilate)
        for i, f in enumerate(mids):
            self.proc(f, float(t0 + (i + 1) * dt_fine))
        self.proc(cur["hdr"], float(t0 + k * dt_fine))

    def drain(self):
        return self.sink.drain()

    def reference_state(self):
        """Package-internal per-pixel log reference (H,W) as numpy, or None before initialization."""
        r = self.proc.ref_log_intensity
        return None if r is None else r[0].detach().cpu().numpy().copy()

    def get_state(self):
        return dict(proc=copy.deepcopy(self.proc.__dict__ | {"recorder": None}), initialized=self.initialized,
                    pending=[dict(c) for c in self.sink.chunks])

    def set_state(self, s):
        d = copy.deepcopy(s["proc"])
        d["recorder"] = self.sink
        self.proc.__dict__.update(d)
        self.sink.chunks = [dict(c) for c in s["pending"]]
        self.initialized = s["initialized"]

    def describe(self):
        d = package_info()
        d["configured"] = dict(threshold_log_units=self.cfg.event_threshold, symmetric=True,
                               event_source=self.cfg.event_source, mode=self.cfg.event_mode,
                               noise=dataclasses.asdict(self.proc.noise.cfg) if self.proc.noise else None,
                               reset_method="new BatchedMultiCamProcessor per episode",
                               polarity="+1 brighter, -1 darker")
        return d


def count_maps(events, width, height):
    """(2,H,W) int32: channel 0 positive, channel 1 negative counts."""
    m = np.zeros((2, height, width), np.int32)
    pos = events["p"] > 0
    np.add.at(m[0], (events["y"][pos], events["x"][pos]), 1)
    np.add.at(m[1], (events["y"][~pos], events["x"][~pos]), 1)
    return m


def voxel_grid(events, width, height, t0, t1, bins=5):
    """(2,bins,H,W) float32 polarity-separated counts in equal time bins over ``(t0, t1]``."""
    g = np.zeros((2, bins, height, width), np.float32)
    if len(events["t"]) == 0:
        return g
    b = np.clip(np.ceil((events["t"] - t0) / (t1 - t0) * bins).astype(int) - 1, 0, bins - 1)
    c = (events["p"] < 0).astype(int)
    np.add.at(g, (c, b, events["y"], events["x"]), 1.0)
    return g
