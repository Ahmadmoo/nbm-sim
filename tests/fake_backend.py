"""Analytic test backend: renders the ``textured_plane`` scene's front face (world y = 1.0) by
ray-plane intersection. Used only by the pure test-suite to exercise clocking, packets and
recording around the real EVIS processor; it is not part of the simulator."""
import numpy as np
import torch


class PlaneBackend:
    def __init__(self, cfg, plane_y=1.0, panel=((-1.0, 1.0), (0.2, 2.2))):
        self.cfg, self.y, self.panel = cfg, plane_y, panel
        self.K = cfg.K.copy()
        u, v = np.meshgrid(np.arange(cfg.width), np.arange(cfg.height))
        self.rays = np.stack([u, v, np.ones_like(u)], -1).astype(np.float64) @ np.linalg.inv(self.K).T
        self.captures = []

    def _texture(self, x, z):
        return 0.15 + 0.35 * (1 + np.sin(9.0 * x) * np.cos(7.0 * z)) + 0.1 * np.sin(31.0 * x + 17.0 * z)

    def capture(self, T_wc, channels):
        self.captures.append((T_wc.copy(), tuple(channels)))
        R, p = T_wc[:3, :3], T_wc[:3, 3]
        d = self.rays @ R.T
        s = (self.y - p[1]) / d[..., 1]
        x, z = p[0] + s * d[..., 0], p[2] + s * d[..., 2]
        (x0, x1), (z0, z1) = self.panel
        hit = (s > 0) & (x >= x0) & (x <= x1) & (z >= z0) & (z <= z1)
        I = np.where(hit, self._texture(x, z), 0.05)
        out = {"T_wc_readback": T_wc.copy()}
        out["hdr_torch"] = torch.tensor(np.repeat(I[None, ..., None], 3, -1), dtype=torch.float32)
        out["rgb"] = (np.clip(np.repeat(I[..., None], 3, -1), 0, 1) * 255).astype(np.uint8)
        depth = np.where(hit, s, np.nan).astype(np.float32)
        out["depth"], out["depth_valid"], out["mask"] = depth, hit, hit
        out["depth_t"] = torch.tensor(np.nan_to_num(depth, nan=1e4)[None])
        out["mv"] = torch.zeros((1, self.cfg.height, self.cfg.width, 2))   # scheduling tests only
        return out

    def describe(self):
        return dict(backend="analytic plane (tests only)")

    def close(self):
        pass
