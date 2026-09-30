import numpy as np

from nbm_sim.geometry import (K_isaac_to_public, K_public_to_isaac, look_at_T_wc, optical_to_gl, project,
                              quat_wxyz_to_rotmat, rotmat_to_quat_wxyz, se3_exp, so3_exp, so3_log)


def test_se3_pure_translation_and_rotation():
    T = se3_exp([0.1, -0.2, 0.3, 0, 0, 0])
    assert np.allclose(T[:3, 3], [0.1, -0.2, 0.3]) and np.allclose(T[:3, :3], np.eye(3))
    th = 0.7
    T = se3_exp([0, 0, 0, 0, 0, th])
    assert np.allclose(T[:3, :3], [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])


def test_se3_screw_motion_matches_closed_form():
    # constant body twist: forward speed v along x while yawing at w about z -> circle of radius v/w
    v, w, t = 0.2, 0.5, 1.3
    T = se3_exp(t * np.array([v, 0, 0, 0, 0, w]))
    r = v / w
    assert np.allclose(T[:3, 3], [r * np.sin(w * t), r * (1 - np.cos(w * t)), 0], atol=1e-12)


def test_so3_log_roundtrip_incl_near_pi():
    rng = np.random.default_rng(0)
    for phi in list(rng.normal(size=(50, 3))) + [np.array([np.pi - 1e-8, 0, 0]), np.array([0, 0, 1e-10])]:
        n = np.linalg.norm(phi)
        if n > np.pi:
            phi = phi / n * (2 * np.pi - n) * -1
        assert np.allclose(so3_exp(so3_log(so3_exp(phi))), so3_exp(phi), atol=1e-7)


def test_look_at_axes():
    T = look_at_T_wc((0, 0, 1), (0, 1, 1))
    R = T[:3, :3]
    assert np.allclose(R[:, 0], [1, 0, 0]) and np.allclose(R[:, 1], [0, 0, -1]) and np.allclose(R[:, 2], [0, 1, 0])
    assert np.isclose(np.linalg.det(R), 1)


def test_optical_to_gl_basis():
    T = look_at_T_wc((0, 0, 1), (0, 1, 1))
    G = optical_to_gl(T)
    assert np.allclose(G[:3, 2], [0, -1, 0])       # USD camera looks along -Z
    assert np.allclose(G[:3, 1], [0, 0, 1])        # +Y up


def test_quat_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(100):
        R = so3_exp(rng.normal(size=3) * 2)
        assert np.allclose(quat_wxyz_to_rotmat(rotmat_to_quat_wxyz(R)), R, atol=1e-10)


def test_projection_and_principal_point_convention():
    from nbm_sim.config import SimConfig
    K = SimConfig().K
    assert np.isclose(K[0, 0], 554.2562584220408) and K[0, 2] == 319.5 and K[1, 2] == 239.5
    assert np.allclose(K_isaac_to_public(K_public_to_isaac(K)), K)
    assert K_public_to_isaac(K)[0, 2] == 320.0
    T = look_at_T_wc((0, 0, 1), (0, 1, 1))
    uv, z = project(K, T, np.array([[0, 2, 1], [0.1, 1, 1]]))
    assert np.allclose(uv[0], [319.5, 239.5]) and np.isclose(z[0], 2)
    assert uv[1, 0] > 319.5   # world +x is image right for this pose
