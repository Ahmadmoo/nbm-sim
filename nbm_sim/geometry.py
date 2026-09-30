"""SE(3)/SO(3) helpers and frame conventions (spec §5).

Public pose convention: ``T_wc`` maps camera optical coordinates (+X right, +Y down,
+Z forward) to world coordinates (right-handed, +Z up), ``P_w = R_wc P_c + t_wc``.
"""
from __future__ import annotations

import numpy as np

# optical (x right, y down, z forward) -> USD/OpenGL camera (x right, y up, -z forward)
OPTICAL_TO_GL = np.diag([1.0, -1.0, -1.0])


def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def so3_exp(phi):
    phi = np.asarray(phi, dtype=np.float64)
    th = np.linalg.norm(phi)
    W = skew(phi)
    if th < 1e-8:
        return np.eye(3) + W + 0.5 * W @ W
    return np.eye(3) + np.sin(th) / th * W + (1 - np.cos(th)) / th**2 * W @ W


def so3_log(R):
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-8:
        return 0.5 * v
    if np.pi - th < 1e-6:
        M = R + np.eye(3)
        a = M[:, int(np.argmax(np.linalg.norm(M, axis=0)))]
        a = a / np.linalg.norm(a)
        return th * (a if np.dot(a, v) >= 0 else -a)
    return th / (2.0 * np.sin(th)) * v


def se3_exp(xi):
    """Exp of body twist ``xi = [v, w]`` (already scaled by duration)."""
    xi = np.asarray(xi, dtype=np.float64)
    v, w = xi[:3], xi[3:]
    th = np.linalg.norm(w)
    W = skew(w)
    if th < 1e-8:
        V = np.eye(3) + 0.5 * W + W @ W / 6.0
    else:
        V = np.eye(3) + (1 - np.cos(th)) / th**2 * W + (th - np.sin(th)) / th**3 * W @ W
    T = np.eye(4)
    T[:3, :3] = so3_exp(w)
    T[:3, 3] = V @ v
    return T


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def look_at_R_wc(position, target, world_up=(0.0, 0.0, 1.0)):
    """Optical-frame rotation looking from ``position`` at ``target`` with zero roll."""
    z = np.asarray(target, float) - np.asarray(position, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(world_up, float))
    if np.linalg.norm(x) < 1e-9:
        raise ValueError("look direction is parallel to world up")
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def look_at_T_wc(position, target, world_up=(0.0, 0.0, 1.0)):
    return make_T(look_at_R_wc(position, target, world_up), np.asarray(position, float))


def optical_to_gl(T_wc):
    """Pose of the USD/OpenGL camera prim for an optical-frame pose."""
    T = T_wc.copy()
    T[:3, :3] = T_wc[:3, :3] @ OPTICAL_TO_GL
    return T


def rotmat_to_quat_wxyz(R):
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat_wxyz_to_rotmat(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_to_order(q_wxyz, order):
    return np.asarray(q_wxyz) if order == "wxyz" else np.roll(q_wxyz, -1)


def quat_from_order(q, order):
    return np.asarray(q) if order == "wxyz" else np.roll(q, 1)


def project(K, T_wc, P_w):
    """Pixel coordinates (pixel-center convention) and optical Z of world points (N,3)."""
    P_c = (np.asarray(P_w) - T_wc[:3, 3]) @ T_wc[:3, :3]
    uv = P_c @ K.T
    return uv[:, :2] / uv[:, 2:3], P_c[:, 2]


def K_public_to_isaac(K):
    """Isaac/USD places the principal point at W/2 in pixel-edge coordinates."""
    Ki = K.copy()
    Ki[0, 2] += 0.5
    Ki[1, 2] += 0.5
    return Ki


def K_isaac_to_public(Ki):
    K = np.asarray(Ki, dtype=np.float64).copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def rotation_angle_between(Ra, Rb):
    return float(np.linalg.norm(so3_log(Ra.T @ Rb)))
