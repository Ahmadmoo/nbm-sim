"""Option C planner: score each motion primitive on the voting volume, pick the best.

Score of a candidate path (only voxels whose edge exists but whose depth is still unsure):

    gain(v) = spread_after(v) - spread_before(v)

where the "after" ray set adds, for each pair of consecutive predicted poses in which the voxel is
visible and fires, a ray from the camera center to the voxel. A voxel fires when its image motion
crosses its projected edge direction (|sin| of the angle between them); for surface-like voxels
(low linearity) the edge direction is undefined and any motion counts. The added mass is the voxel's
current vote count times the fraction of the path in which it fires.

Limits: no occlusion test (back-side voxels can be counted visible); brightness-gradient magnitude is
not modeled. ``branch_gains`` measures the true gain so the score can be checked (spec §12 oracle,
labeled as such).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .evaluation import reconstruction_metrics
from .primitives import instantiate, predict_path
from .voting import EDGE_UNSURE, FIX, UNOBSERVED


def _project(P, Ts, K):
    """P (M,3), Ts (S,4,4) -> uv (S,M,2), z (S,M)."""
    pc = torch.einsum("sij,smi->smj", Ts[:, :3, :3], P[None] - Ts[:, None, :3, 3])
    z = pc[..., 2]
    uv = torch.stack([K[0, 0] * pc[..., 0] / z + K[0, 2], K[1, 1] * pc[..., 1] / z + K[1, 2]], -1)
    return uv, z


def observability_gain(volume, poses, K, width, height, u_min_px=0.25, w_unobserved=0.0, near=0.05,
                       coverage_stride=7):
    a = volume.analyze()
    dev = volume.device
    Ts = torch.tensor(np.asarray(poses), device=dev, dtype=torch.float64)
    Kt = torch.tensor(np.asarray(K), device=dev, dtype=torch.float64)
    sel = a["state"][a["ridge_idx"]] == EDGE_UNSURE
    ridx = a["ridge_idx"][sel]
    out = dict(n_unsure=int(len(ridx)), edge_gain=0.0, edge_gain_mean=0.0, coverage=0.0)

    def in_image(uv, z):
        return (z > near) & (uv[..., 0] >= -0.5) & (uv[..., 0] <= width - 0.5) & \
            (uv[..., 1] >= -0.5) & (uv[..., 1] <= height - 0.5)

    if len(ridx) and len(Ts) > 1:
        P = volume.centers[ridx]
        e, lin = a["edge_dir"][sel], a["linearity"][sel]
        n_old = volume.votes[ridx].double()
        S_old = volume.dir_sum[ridx].double() / FIX
        uv, z = _project(P, Ts, Kt)
        uv_e, _ = _project(P + volume.voxel * e, Ts, Kt)
        vis = in_image(uv, z)
        u = uv[1:] - uv[:-1]
        speed = u.norm(dim=-1)
        eh = uv_e[:-1] - uv[:-1]
        eh = eh / eh.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sin = (u[..., 0] * eh[..., 1] - u[..., 1] * eh[..., 0]).abs() / speed.clamp(min=1e-12)
        fires = (vis[1:] & vis[:-1] & (speed > u_min_px)).double() * (lin * sin + (1.0 - lin))
        dnew = P[None] - Ts[:-1, None, :3, 3]
        dnew = dnew / dnew.norm(dim=-1, keepdim=True)
        npair = fires.shape[0]
        add_vec = n_old[:, None] * (fires[..., None] * dnew).sum(0) / npair
        add_n = n_old * fires.sum(0) / npair
        r_old = S_old.norm(dim=1) / n_old
        r_new = (S_old + add_vec).norm(dim=1) / (n_old + add_n)
        gain = (r_old - r_new).clamp(min=0.0)
        out["edge_gain"] = float(gain.sum())
        out["edge_gain_mean"] = float(gain.mean())
    if w_unobserved > 0:
        uidx = (a["state"] == UNOBSERVED).nonzero(as_tuple=True)[0][::coverage_stride]
        if len(uidx):
            uv, z = _project(volume.centers[uidx], Ts, Kt)
            out["coverage"] = float(in_image(uv, z).any(0).double().mean())
    out["score"] = out["edge_gain_mean"] + w_unobserved * out["coverage"]
    return out


class ObservabilityPlanner:
    """Chooses the next primitive by ``observability_gain``. Feasibility uses the same motion
    controller and safety model as execution (declared prior); scoring uses only the voting volume."""

    def __init__(self, library, speed=0.12, move_s=0.6, stop_s=0.4, sample_every=0.05, w_unobserved=0.0):
        self.library = list(library)
        self.kw = dict(speed=speed, move_s=move_s, stop_s=stop_s)
        self.sample_every, self.w_unobserved = sample_every, w_unobserved

    def schedule(self, prim, env):
        return instantiate(prim, env.camera_state(), env.cfg, **self.kw)

    def score_all(self, env, volume):
        rows = []
        for prim in self.library:
            sched = self.schedule(prim, env)
            pred = predict_path(env, sched, self.sample_every)
            row = dict(name=prim.name, feasible=pred["feasible"] and pred["at_rest"],
                       reason=pred["reason"] or ("" if pred["at_rest"] else "not_at_rest"))
            if row["feasible"]:
                row.update(observability_gain(volume, pred["poses"], env.K, env.cfg.width, env.cfg.height,
                                              w_unobserved=self.w_unobserved))
                row.update(path_length=pred["path_length"], rotation=pred["rotation"])
            else:
                row["score"] = -math.inf
            rows.append((prim, sched, row))
        return rows

    def choose(self, env, volume):
        """Best feasible primitive (first one wins ties). Returns (primitive, schedule, rows) or None."""
        rows = self.score_all(env, volume)
        best = max(range(len(rows)), key=lambda i: (rows[i][2]["score"], -i))
        if rows[best][2]["score"] == -math.inf:
            return None
        return rows[best][0], rows[best][1], [r[2] for r in rows]


def oracle_fscore(env, tau=0.005, surface="accessible_surface", margin=0.02):
    """Evaluator-only (privileged) F-score of the volume's points against the target surface."""
    geom = env.get_evaluation_state()["geometry"]
    region = env.spec.task_region(margin)
    key = f"fscore@{tau * 1e3:g}mm"

    def evaluate(volume, log=None):
        return reconstruction_metrics(volume.points()["xyz"], geom[surface], region, taus=(tau,))[key]
    return evaluate


def branch_gains(env, volume, planner, evaluate, log=None):
    """Oracle: from the current state, run every feasible primitive for real, measure
    ``evaluate(volume_after, log_after) - evaluate(volume, log)``, and restore the state.
    Branch packets are not recorded. Returns (base_value, rows)."""
    base = env.get_state()
    keep = dict(recorder=env.recorder, counts=dict(env.counts), timing=dict(env.timing),
                last_eval=env._last_eval, last_intensity=getattr(env, "_last_intensity", None))
    env.recorder = None
    base_value = evaluate(volume, log)
    rows = []
    try:
        for prim in planner.library:
            sched = planner.schedule(prim, env)
            pred = predict_path(env, sched, planner.sample_every)
            if not (pred["feasible"] and pred["at_rest"]):
                rows.append(dict(name=prim.name, feasible=False, gain=math.nan))
                continue
            vb, lb, cut = volume.copy(), (log.copy() if log is not None else None), False
            for v, d in sched:
                p = env.step(v, d)
                vb.update(p)
                if lb is not None:
                    lb.add(p)
                if p["terminated"] or p["truncated"]:
                    cut = True
                    break
            value = evaluate(vb, lb)
            rows.append(dict(name=prim.name, feasible=True, value=value, gain=value - base_value, cut=cut))
            env.set_state(base)
            env.counts, env.timing = dict(keep["counts"]), dict(keep["timing"])
            env._last_eval, env._last_intensity = keep["last_eval"], keep["last_intensity"]
    finally:
        env.recorder = keep["recorder"]
    return base_value, rows


def rank_agreement(score_rows, gain_rows):
    """Spearman rank correlation between predicted scores and true gains over feasible primitives."""
    from scipy.stats import spearmanr
    g = {r["name"]: r["gain"] for r in gain_rows if r["feasible"] and np.isfinite(r["gain"])}
    names = [r["name"] for r in score_rows if r["feasible"] and r["name"] in g]
    if len(names) < 3:
        return dict(rho=math.nan, n=len(names))
    s = [next(r["score"] for r in score_rows if r["name"] == n) for n in names]
    rho = spearmanr(s, [g[n] for n in names]).statistic
    gains = np.array([g[n] for n in names])
    return dict(rho=float(rho), n=len(names), gain_spread=float(gains.max() - gains.min()),
                best_true=names[int(np.argmax(gains))], best_pred=names[int(np.argmax(s))])
