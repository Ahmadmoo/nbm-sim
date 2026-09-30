"""Packet contract (spec §11.2), schema checks, and observation-protocol filtering (§10.2)."""
from __future__ import annotations

import numpy as np

SCHEMA_VERSION = "1.0"

EVENT_DTYPES = dict(x=np.uint16, y=np.uint16, t=np.float64, p=np.int8)

# fields every protocol may see; privileged evaluation data is never placed in a packet
_BASE = {"schema_version", "episode_id", "step_index", "phase", "t_start", "t_end", "events", "K", "image_size",
         "T_wc_start", "T_wc_end", "pose_t", "pose_T_wc", "requested_velocity", "executed_twist",
         "command_limited", "safety_intervention", "terminated", "truncated", "reason", "diagnostics"}
PROTOCOL_FIELDS = {
    "events_only": _BASE,
    "events_rgb_known_pose": _BASE | {"rgb", "rgb_t", "rgb_T_wc"},
    "events_rgbd_known_pose": _BASE | {"rgb", "rgb_t", "rgb_T_wc", "depth_observed", "depth_observed_t"},
    "oracle": _BASE | {"rgb", "rgb_t", "rgb_T_wc", "intensity", "depth_observed", "depth_observed_t"},
}


def frozen(a):
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


def empty_events():
    return {k: frozen(np.zeros(0, dtype=d)) for k, d in EVENT_DTYPES.items()}


def concat_events(chunks):
    """Concatenate event chunks and sort by time with a stable tie rule (package emission order)."""
    if not chunks:
        return empty_events()
    ev = {k: np.concatenate([c[k] for c in chunks]).astype(d) for k, d in EVENT_DTYPES.items()}
    order = np.argsort(ev["t"], kind="stable")
    return {k: frozen(v[order]) for k, v in ev.items()}


def actor_view(packet, protocol, expose_intensity=False):
    allowed = PROTOCOL_FIELDS[protocol]
    if expose_intensity and protocol != "events_only":
        allowed = allowed | {"intensity"}
    return {k: v for k, v in packet.items() if k in allowed}


def validate_packet(p, width, height):
    """Return a list of contract violations (empty list = valid)."""
    err = []
    ev = p["events"]
    n = len(ev["t"])
    for k, d in EVENT_DTYPES.items():
        if ev[k].dtype != d or ev[k].shape != (n,):
            err.append(f"events.{k} dtype/shape")
    if n:
        if np.any(np.diff(ev["t"]) < 0):
            err.append("events not sorted by time")
        if not (p["t_start"] < ev["t"].min() and ev["t"].max() <= p["t_end"] + 1e-12):
            err.append("event time outside (t_start, t_end]")
        if ev["x"].max() >= width or ev["y"].max() >= height:
            err.append("event coordinate out of bounds")
        if not np.all(np.isin(ev["p"], (-1, 1))):
            err.append("polarity not in {-1,+1}")
    J = len(p["pose_t"])
    if p["pose_T_wc"].shape != (J, 4, 4) or p["executed_twist"].shape != (max(J - 1, 0), 6):
        err.append("pose/twist shapes")
    if "rgb" in p:
        M = p["rgb"].shape[0]
        if p["rgb"].dtype != np.uint8 or p["rgb"].shape[1:] != (height, width, 3):
            err.append("rgb dtype/shape")
        if p["rgb_t"].shape != (M,) or p["rgb_T_wc"].shape != (M, 4, 4):
            err.append("rgb_t/rgb_T_wc shapes")
    if p["K"].shape != (3, 3) or list(p["image_size"]) != [width, height]:
        err.append("K/image_size")
    return err
