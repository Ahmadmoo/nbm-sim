"""Ray-voting volume: the fast, deterministic in-loop map for option C.

Every event casts a ray from the camera center (known pose at the event time) through its pixel.
Rays vote into a world-frame voxel grid around the target; real edges are where rays from different
viewpoints cross (the EMVS idea, here in one world grid instead of per-keyframe volumes).

Per voxel the grid keeps three integer counters, so updates are order-independent and exactly
repeatable on CPU and GPU:

* ``votes``   number of event rays through the voxel
* ``dir_sum`` sum of the rays' unit directions in fixed point (angle spread = how well depth is fixed)
* ``seen``    number of updates in which the voxel center was inside the camera frustum (no occlusion test)

Points are extracted as in EMVS: from keyframe reference views, each pixel ray takes the voxel with
the most votes along it (middle of ties), and pixels whose confidence beats a global ratio and their
local mean are kept. Depth along a pixel ray is only as precise as voxel * depth / baseline, which is
what the per-voxel angle spread measures. Voting needs sharp image edges; smooth texture gives no
peak at the surface.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation

FIX = 2 ** 20                      # fixed-point scale of dir_sum
UNOBSERVED, SEEN, EDGE_UNSURE, EDGE_KNOWN = 0, 1, 2, 3


def event_poses(t, pose_t, pose_T):
    """Rotation (N,3,3) and center (N,3) at each event time. Exact at pose samples (nbm-sim ticks);
    linear/slerp interpolation otherwise."""
    pose_t = np.asarray(pose_t, np.float64)
    idx = np.clip(np.searchsorted(pose_t, t), 0, len(pose_t) - 1)
    R, c = pose_T[idx, :3, :3].copy(), pose_T[idx, :3, 3].copy()
    off = np.abs(pose_t[idx] - t) > 1e-9
    if off.any():
        i1 = np.clip(idx[off], 1, len(pose_t) - 1)
        i0 = i1 - 1
        a = ((t[off] - pose_t[i0]) / (pose_t[i1] - pose_t[i0]))[:, None]
        c[off] = (1 - a) * pose_T[i0, :3, 3] + a * pose_T[i1, :3, 3]
        r0 = Rotation.from_matrix(pose_T[i0, :3, :3])
        rel = (r0.inv() * Rotation.from_matrix(pose_T[i1, :3, :3])).as_rotvec()
        R[off] = (r0 * Rotation.from_rotvec(rel * a)).as_matrix()
    return R, c


class VotingVolume:
    """Incremental ray-voting map. Implements ``Reconstructor.update(packet) -> map_state``."""

    def __init__(self, lo, hi, voxel=0.002, device="cpu", step_ratio=0.5, near=0.05, chunk=8192,
                 conf_ratio=0.5, min_votes=5, local_win=5, ref_stride=2, kf_dist=0.03, kf_angle_deg=5.0,
                 max_refs=None, min_angle_deg=10.0, edge_radius=2):
        self.lo, self.hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        self.voxel, self.device, self.near, self.chunk = float(voxel), torch.device(device), near, chunk
        self.step = self.voxel * step_ratio
        self.shape = tuple(int(math.ceil(s)) for s in (self.hi - self.lo) / self.voxel)
        self.params = dict(conf_ratio=conf_ratio, min_votes=min_votes, local_win=local_win, ref_stride=ref_stride,
                           kf_dist=kf_dist, kf_angle_deg=kf_angle_deg, max_refs=max_refs,
                           min_angle_deg=min_angle_deg, edge_radius=edge_radius)
        self.keyframes = []
        n = int(np.prod(self.shape))
        self.votes = torch.zeros(n, dtype=torch.int64, device=self.device)
        self.dir_sum = torch.zeros((n, 3), dtype=torch.int64, device=self.device)
        self.seen = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.n_events = 0
        self.n_updates = 0
        self._centers = None
        self._cache = None

    @classmethod
    def for_scene(cls, spec, voxel=0.002, margin=0.05, **kw):
        lo, hi = spec.task_region(margin)
        return cls(lo, hi, voxel, **kw)

    # ---------- geometry helpers ----------
    @property
    def centers(self):
        if self._centers is None:
            g = [torch.arange(s, device=self.device, dtype=torch.float64) for s in self.shape]
            ijk = torch.stack(torch.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
            self._centers = torch.tensor(self.lo, device=self.device) + (ijk + 0.5) * self.voxel
        return self._centers

    def _flat(self, ijk):
        return (ijk[..., 0] * self.shape[1] + ijk[..., 1]) * self.shape[2] + ijk[..., 2]

    def copy(self):
        v = VotingVolume.__new__(VotingVolume)
        v.__dict__.update(self.__dict__)
        v.votes, v.dir_sum, v.seen = self.votes.clone(), self.dir_sum.clone(), self.seen.clone()
        v.keyframes = list(self.keyframes)
        v._cache = None
        return v

    # ---------- update ----------
    def update(self, packet):
        ev = packet["events"]
        K = np.asarray(packet["K"], np.float64)
        W, H = (int(v) for v in packet["image_size"])
        if len(ev["t"]):
            R, c = event_poses(np.asarray(ev["t"], np.float64), packet["pose_t"], packet["pose_T_wc"])
            pix = np.stack([ev["x"].astype(np.float64), ev["y"].astype(np.float64), np.ones(len(ev["t"]))], 1)
            d = np.einsum("nij,nj->ni", R, pix @ np.linalg.inv(K).T)
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            for s in range(0, len(d), self.chunk):
                self._vote(torch.tensor(c[s:s + self.chunk], device=self.device),
                           torch.tensor(d[s:s + self.chunk], device=self.device))
            self.n_events += len(ev["t"])
        T_end = np.asarray(packet["T_wc_end"], np.float64)
        self._mark_seen(T_end, K, W, H)
        self._add_keyframe(T_end, K, W, H)
        self.n_updates += 1
        self._cache = None
        return self

    def _march(self, o, d):
        """Sample rays inside the box: flat voxel index (n,L) and inside-mask (n,L)."""
        lo = torch.tensor(self.lo, device=self.device)
        hi = torch.tensor(self.hi, device=self.device)
        inv = 1.0 / torch.where(d >= 0, d.clamp(min=1e-12), d.clamp(max=-1e-12))
        t0, t1 = (lo - o) * inv, (hi - o) * inv
        tmin = torch.minimum(t0, t1).amax(1).clamp(min=self.near)
        tmax = torch.maximum(t0, t1).amin(1)
        span = (tmax - tmin).clamp(min=0.0)
        L = max(1, int(math.ceil(float(span.max()) / self.step)))
        s = tmin[:, None] + (torch.arange(L, device=self.device, dtype=torch.float64) + 0.5) * self.step
        inside = (s < tmax[:, None]) & (tmax > tmin)[:, None]
        ijk = torch.floor((o[:, None, :] + s[..., None] * d[:, None, :] - lo) / self.voxel).long()
        for a in range(3):
            ijk[..., a] = ijk[..., a].clamp(0, self.shape[a] - 1)
        return self._flat(ijk), inside

    def _vote(self, o, d):
        flat, inside = self._march(o, d)
        if not inside.any():
            return
        keep = inside.clone()
        keep[:, 1:] &= flat[:, 1:] != flat[:, :-1]               # one vote per voxel per ray
        rows = keep.nonzero(as_tuple=True)[0]
        idx = flat[keep]
        dq = torch.round(d * FIX).long()
        self.votes.index_add_(0, idx, torch.ones_like(idx))
        self.dir_sum.index_add_(0, idx, dq[rows])

    def _mark_seen(self, T_wc, K, W, H):
        R = torch.tensor(T_wc[:3, :3], device=self.device)
        c = torch.tensor(T_wc[:3, 3], device=self.device)
        pc = (self.centers - c) @ R
        z = pc[:, 2]
        u = K[0, 0] * pc[:, 0] / z + K[0, 2]
        v = K[1, 1] * pc[:, 1] / z + K[1, 2]
        vis = (z > self.near) & (u >= -0.5) & (u <= W - 0.5) & (v >= -0.5) & (v <= H - 0.5)
        self.seen += vis.to(torch.int32)

    def _add_keyframe(self, T, K, W, H):
        if self.keyframes:
            Tk = self.keyframes[-1][0]
            moved = np.linalg.norm(T[:3, 3] - Tk[:3, 3])
            turned = math.degrees(np.linalg.norm(Rotation.from_matrix(Tk[:3, :3].T @ T[:3, :3]).as_rotvec()))
            if moved < self.params["kf_dist"] and turned < self.params["kf_angle_deg"]:
                return
        self.keyframes.append((T.copy(), K.copy(), W, H))

    def _extract_view(self, T, K, W, H):
        """EMVS step for one reference view: best voxel along each pixel ray, adaptive threshold."""
        p = self.params
        st = p["ref_stride"]
        u, v = np.meshgrid(np.arange(0, W, st), np.arange(0, H, st))
        pix = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], 1).astype(np.float64)
        d = (pix @ np.linalg.inv(K).T) @ T[:3, :3].T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        conf = torch.zeros(len(d), dtype=torch.float64, device=self.device)
        best = torch.zeros(len(d), dtype=torch.int64, device=self.device)
        o = torch.tensor(T[:3, 3], device=self.device)
        for s in range(0, len(d), self.chunk):
            dc = torch.tensor(d[s:s + self.chunk], device=self.device)
            flat, inside = self._march(o.expand(len(dc), 3), dc)
            val = torch.where(inside, self.votes[flat].double(), torch.full(flat.shape, -1.0, dtype=torch.float64,
                                                                            device=self.device))
            m = val.max(1).values
            ties = (val == m[:, None]) & inside
            pos = torch.arange(val.shape[1], device=self.device, dtype=torch.float64)
            mid = torch.round((ties * pos).sum(1) / ties.sum(1).clamp(min=1)).long()     # middle of ties
            conf[s:s + self.chunk] = m.clamp(min=0.0)
            best[s:s + self.chunk] = flat.gather(1, mid[:, None])[:, 0]
        cm = conf.reshape(u.shape)
        pos = cm[cm > 0]
        if len(pos) == 0:
            return best[:0]
        thr = max(float(p["min_votes"]), p["conf_ratio"] * float(torch.quantile(pos, 0.99)))
        w = p["local_win"]
        local = torch.nn.functional.avg_pool2d(cm[None, None], w, 1, w // 2, count_include_pad=False)[0, 0]
        ok = (cm >= thr) & (cm > local)
        return best[ok.reshape(-1)]

    # ---------- analysis ----------
    def analyze(self):
        """Ridge voxels, angle spread, states, and edge direction/linearity for ridge voxels (cached)."""
        if self._cache is not None:
            return self._cache
        p = self.params
        votes = self.votes.double()
        refs = self.keyframes if p["max_refs"] is None else self.keyframes[-p["max_refs"]:]
        ridge = torch.zeros(len(votes), dtype=torch.bool, device=self.device)
        for T, K, W, H in refs:
            ridge[self._extract_view(T, K, W, H)] = True
        rbar = (self.dir_sum.double().norm(dim=1) / FIX) / votes.clamp(min=1)
        spread = (1.0 - rbar).clamp(min=0.0)
        angle_deg = torch.rad2deg(torch.sqrt(2.0 * spread))       # small-angle std of ray directions
        state = torch.full_like(self.seen, SEEN, dtype=torch.int8)
        state[self.seen == 0] = UNOBSERVED
        state[ridge & (angle_deg < p["min_angle_deg"])] = EDGE_UNSURE
        state[ridge & (angle_deg >= p["min_angle_deg"])] = EDGE_KNOWN
        ridx = ridge.nonzero(as_tuple=True)[0]
        edge_dir, linearity = self._edge_structure(ridx, votes * ridge)
        self._cache = dict(n_refs=len(refs), ridge=ridge, spread=spread, angle_deg=angle_deg, state=state,
                           ridge_idx=ridx, edge_dir=edge_dir, linearity=linearity)
        return self._cache

    def _edge_structure(self, ridx, w):
        """Principal direction and linearity (l1-l2)/l1 of ridge voxels in a (2r+1)^3 window, using offsets
        projected perpendicular to each voxel's mean viewing direction (depth scatter of unsure voxels
        would otherwise turn a line into a sheet)."""
        if len(ridx) == 0:
            return torch.zeros((0, 3), dtype=torch.float64, device=self.device), \
                torch.zeros(0, dtype=torch.float64, device=self.device)
        r = self.params["edge_radius"]
        g = torch.arange(-r, r + 1, device=self.device)
        off = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        sz = torch.tensor(self.shape, device=self.device)
        ijk = torch.stack([ridx // (self.shape[1] * self.shape[2]), (ridx // self.shape[2]) % self.shape[1],
                           ridx % self.shape[2]], 1)
        m = self.dir_sum[ridx].double()
        m = m / m.norm(dim=1, keepdim=True).clamp(min=1e-12)          # mean viewing direction
        cov = torch.zeros((len(ridx), 3, 3), dtype=torch.float64, device=self.device)
        for s in range(0, len(ridx), 4096):
            nb = ijk[s:s + 4096, None, :] + off[None]
            inb = ((nb >= 0) & (nb < sz)).all(-1)
            wt = w[self._flat(nb.clamp(min=0) % sz)] * inb
            ms = m[s:s + 4096]
            o = off.double()[None] - (off.double()[None] @ ms[:, :, None]) * ms[:, None, :]   # drop the depth part
            cov[s:s + 4096] = torch.einsum("nk,nki,nkj->nij", wt, o, o) / wt.sum(1).clamp(min=1e-12)[:, None, None]
        lam, vec = torch.linalg.eigh(cov)
        linearity = ((lam[:, 2] - lam[:, 1]) / lam[:, 2].clamp(min=1e-12)).clamp(0, 1)
        return vec[:, :, 2], linearity

    def points(self):
        """Ridge voxel centers with their attributes (the partial 3D reconstruction)."""
        a = self.analyze()
        i = a["ridge_idx"]
        return dict(xyz=self.centers[i].cpu().numpy(), votes=self.votes[i].cpu().numpy(),
                    angle_deg=a["angle_deg"][i].cpu().numpy(), state=a["state"][i].cpu().numpy())

    def summary(self):
        a = self.analyze()
        st = a["state"]
        return dict(n_events=self.n_events, n_updates=self.n_updates, n_keyframes=len(self.keyframes),
                    n_unobserved=int((st == UNOBSERVED).sum()), n_seen=int((st == SEEN).sum()),
                    n_edge_unsure=int((st == EDGE_UNSURE).sum()), n_edge_known=int((st == EDGE_KNOWN).sum()))
