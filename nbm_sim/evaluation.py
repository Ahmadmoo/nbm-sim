"""Evaluator-side metrics (spec §15). Uses privileged geometry; never passed to the actor."""
from __future__ import annotations

import numpy as np

from .events import count_maps


def crop_to_region(points, region):
    lo, hi = region
    m = np.all((points >= lo) & (points <= hi), axis=1)
    return points[m], int((~m).sum())


def reconstruction_metrics(points, reference, region, taus=(0.002, 0.005, 0.01)):
    """Target-only accuracy / completeness / precision / recall / F-score.

    ``reference`` are dense surface samples (point-to-sample distance approximates
    point-to-surface within half the sample spacing). Empty reconstructions get
    P = R = F = 0 and infinite distances, by convention.
    """
    from scipy.spatial import cKDTree

    pts, n_out = crop_to_region(np.asarray(points, np.float64).reshape(-1, 3), region)
    ref, _ = crop_to_region(np.asarray(reference, np.float64), region)
    res = dict(n_points=len(pts), n_points_outside_region=n_out, n_reference=len(ref))
    if len(pts) == 0:
        res.update(accuracy_mean=np.inf, accuracy_median=np.inf, completeness_mean=np.inf,
                   completeness_median=np.inf)
        for t in taus:
            res.update({f"precision@{t*1e3:g}mm": 0.0, f"recall@{t*1e3:g}mm": 0.0, f"fscore@{t*1e3:g}mm": 0.0})
        return res
    d_acc = cKDTree(ref).query(pts)[0]
    d_comp = cKDTree(pts).query(ref)[0]
    res.update(accuracy_mean=float(d_acc.mean()), accuracy_median=float(np.median(d_acc)),
               completeness_mean=float(d_comp.mean()), completeness_median=float(np.median(d_comp)))
    for t in taus:
        p, r = float((d_acc < t).mean()), float((d_comp < t).mean())
        res.update({f"precision@{t*1e3:g}mm": p, f"recall@{t*1e3:g}mm": r,
                    f"fscore@{t*1e3:g}mm": 0.0 if p + r == 0 else 2 * p * r / (p + r)})
    return res


def count_discrepancy(a, b, width, height):
    """Per-polarity normalized L1 ``sum|A-B| / max(sum B, 1)`` with B the finer-sampled stream (§17)."""
    A, B = count_maps(a, width, height), count_maps(b, width, height)
    out = {}
    for i, name in enumerate(("pos", "neg")):
        out[f"l1_{name}"] = float(np.abs(A[i] - B[i]).sum() / max(B[i].sum(), 1))
        out[f"count_change_{name}"] = float(abs(int(A[i].sum()) - int(B[i].sum())) / max(int(B[i].sum()), 1))
        out[f"n_{name}_a"], out[f"n_{name}_b"] = int(A[i].sum()), int(B[i].sum())
    return out


def timing_discrepancy(a, b, width, height):
    """Mean |first-event time difference| over pixels that fire in both streams, per polarity."""
    out = {}
    for pol, name in ((1, "pos"), (-1, "neg")):
        first = []
        for e in (a, b):
            m = e["p"] == pol
            ft = np.full(width * height, np.nan)
            idx = e["y"][m].astype(np.int64) * width + e["x"][m]
            order = np.argsort(e["t"][m])[::-1]
            ft[idx[order]] = e["t"][m][order]
            first.append(ft)
        both = np.isfinite(first[0]) & np.isfinite(first[1])
        out[f"first_event_dt_mean_{name}"] = float(np.abs(first[0] - first[1])[both].mean()) if both.any() else None
    return out


def polarity_agreement(a, b, width, height):
    """Fraction of pixels active in both streams whose net signed count has the same sign."""
    A, B = count_maps(a, width, height), count_maps(b, width, height)
    sa, sb = A[0] - A[1], B[0] - B[1]
    both = (A.sum(0) > 0) & (B.sum(0) > 0)
    return float((np.sign(sa[both]) == np.sign(sb[both])).mean()) if both.any() else None


def static_false_event_rate(events, width, height, duration):
    """Events per pixel per second on a stationary camera, plus where they occur."""
    m = count_maps(events, width, height).sum(0)
    ys, xs = np.nonzero(m)
    return dict(rate=float(m.sum() / (width * height * duration)), n_events=int(m.sum()),
                n_active_pixels=int(len(xs)), max_per_pixel=int(m.max()) if m.size else 0,
                active_bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None)


def trajectory_costs(pose_T_wc):
    p = pose_T_wc[:, :3, 3]
    from .geometry import rotation_angle_between
    rot = sum(rotation_angle_between(pose_T_wc[i, :3, :3], pose_T_wc[i + 1, :3, :3])
              for i in range(len(pose_T_wc) - 1))
    return dict(path_length=float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()), rotation_travel=float(rot))


def check_motion_limits(executed_twist, pose_T_wc, dt, cfg, tol=1e-9, start_at_rest=True):
    """Numerical validation of norm speed limits and world-frame acceleration between tick endpoints.
    ``pose_T_wc`` has one more sample than ``executed_twist`` (the start pose)."""
    v, w = executed_twist[:, :3], executed_twist[:, 3:]
    R_end = pose_T_wc[1:, :3, :3]
    v_end = np.einsum("nij,nj->ni", R_end, v)
    w_end = np.einsum("nij,nj->ni", R_end, w)
    if start_at_rest:
        v_end, w_end = np.vstack([np.zeros(3), v_end]), np.vstack([np.zeros(3), w_end])
    a_lin = np.linalg.norm(np.diff(v_end, axis=0), axis=1) / dt if len(v_end) > 1 else np.zeros(0)
    a_ang = np.linalg.norm(np.diff(w_end, axis=0), axis=1) / dt if len(w_end) > 1 else np.zeros(0)
    res = dict(max_speed=float(np.linalg.norm(v, axis=1).max(initial=0)),
               max_angular_speed=float(np.linalg.norm(w, axis=1).max(initial=0)),
               max_linear_acceleration=float(a_lin.max(initial=0)), max_angular_acceleration=float(a_ang.max(initial=0)))
    res["ok"] = bool(res["max_speed"] <= cfg.max_linear_speed * (1 + 1e-9) + tol and
                     res["max_angular_speed"] <= cfg.max_angular_speed * (1 + 1e-9) + tol and
                     res["max_linear_acceleration"] <= cfg.max_linear_acceleration * (1 + 1e-6) + tol and
                     res["max_angular_acceleration"] <= cfg.max_angular_acceleration * (1 + 1e-6) + tol)
    return res
