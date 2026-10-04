"""Option C: voting volume, primitives, observability score, oracle branching, GPERT I/O."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("dvs_gen")

from fake_backend import PlaneBackend
from nbm_sim.config import SimConfig
from nbm_sim.environment import CameraNBMEnv
from nbm_sim.gpert_io import StreamLog, compress_static, export_gpert, gpert_config, load_gpert_points
from nbm_sim.packet import concat_events
from nbm_sim.planner import ObservabilityPlanner, branch_gains
from nbm_sim.primitives import instantiate, make_library, predict_path
from nbm_sim.scene import make_texture
from nbm_sim.voting import EDGE_KNOWN, EDGE_UNSURE, UNOBSERVED, VotingVolume

RNG = np.random.default_rng(3)
RECTS = np.c_[RNG.uniform(-0.6, 0.6, 40), RNG.uniform(0.7, 1.7, 40), RNG.uniform(0.03, 0.15, (40, 2)),
              RNG.uniform(-0.3, 0.3, 40)]


class RectPlane(PlaneBackend):
    def _texture(self, x, z):
        img = np.full(x.shape, 0.4)
        for cx, cz, w, h, a in RECTS:
            img += a * ((np.abs(x - cx) < w / 2) & (np.abs(z - cz) < h / 2))
        return np.clip(img, 0.05, 1.0)


class EdgePlane(PlaneBackend):          # one vertical edge at x = 0
    def _texture(self, x, z):
        return np.where(x < 0.0, 0.2, 0.8)


def make(backend=RectPlane, **kw):
    cfg = SimConfig(scene_id="textured_plane", width=160, height=120, warmup_renders=1, display=False, record=False,
                    device="cpu", observation_protocol="events_only").replace(**kw)
    return CameraNBMEnv(cfg, backend=backend(cfg))


def run(env, vol, names, phase="bootstrap", log=None, **prim_kw):
    lib = {p.name: p for p in make_library()}
    pk = []
    for n in names:
        for v, d in instantiate(lib[n], env.camera_state(), env.cfg, **prim_kw):
            p = env.step(v, d, phase=phase)
            vol.update(p)
            if log is not None:
                log.add(p)
            pk.append(p)
    return pk


def box():
    return (-0.5, 0.8, 0.9), (0.5, 1.2, 1.5)


# ---------- voting volume ----------
def test_votes_peak_on_a_real_edge():
    env = make(EdgePlane, max_linear_acceleration=2.0)
    vol = VotingVolume((-0.3, 0.8, 1.0), (0.3, 1.2, 1.4), 0.005)
    vol.update(env.reset(seed=0))
    for tw in ([0.2, 0, 0, 0, 0, 0], [-0.2, 0, 0, 0, 0, 0], [-0.2, 0, 0, 0, 0, 0], [0.2, 0, 0, 0, 0, 0]):
        for _ in range(10):
            vol.update(env.step(np.array(tw, float)))
    top = vol.centers[vol.votes.argmax()].numpy()
    assert abs(top[0]) <= 0.005 and abs(top[1] - 1.0) <= 0.005


def test_incremental_equals_batch_and_is_deterministic():
    env = make()
    a = VotingVolume(*box(), 0.02)
    pk = [env.reset(seed=0)] + run(env, a, ["slide_0"])
    a2 = VotingVolume(*box(), 0.02)
    for p in pk:
        a2.update(p)
    merged = dict(events=concat_events([p["events"] for p in pk]), K=pk[0]["K"], image_size=pk[0]["image_size"],
                  pose_t=np.concatenate([pk[0]["pose_t"]] + [p["pose_t"][1:] for p in pk[1:]]),
                  pose_T_wc=np.concatenate([pk[0]["pose_T_wc"]] + [p["pose_T_wc"][1:] for p in pk[1:]]),
                  T_wc_end=pk[-1]["T_wc_end"])
    b = VotingVolume(*box(), 0.02).update(merged)
    assert torch.equal(a.votes, a2.votes) and torch.equal(a.dir_sum, a2.dir_sum)
    assert torch.equal(a.votes, b.votes) and torch.equal(a.dir_sum, b.dir_sum)
    assert int(a.votes.sum()) > 0


def test_points_converge_to_the_surface_with_baseline():
    env = make()
    vol = VotingVolume(*box(), 0.01)
    vol.update(env.reset(seed=0))
    run(env, vol, ["slide_0"])
    y1, a1 = vol.points()["xyz"][:, 1], np.median(vol.points()["angle_deg"])
    run(env, vol, ["slide_0", "slide_0", "slide_90"])
    pts = vol.points()
    y = pts["xyz"][:, 1]
    assert abs(np.median(y) - 1.0) <= 0.02
    assert np.mean(np.abs(y - 1.0) <= 0.03) > max(0.4, np.mean(np.abs(y1 - 1.0) <= 0.03))
    assert np.median(pts["angle_deg"]) > a1


def test_states():
    env = make()
    vol = VotingVolume((-1.5, 0.8, 0.9), (1.5, 1.2, 1.5), 0.02)
    vol.update(env.reset(seed=0))
    run(env, vol, ["slide_0"])
    a = vol.analyze()
    st = a["state"]
    assert (st == UNOBSERVED).any()                                   # box wider than the field of view
    edge = (st == EDGE_UNSURE) | (st == EDGE_KNOWN)
    assert torch.equal(edge, a["ridge"]) and edge.any()
    c = vol.copy()
    assert torch.equal(c.votes, vol.votes) and c.votes.data_ptr() != vol.votes.data_ptr()


# ---------- primitives ----------
def test_primitives_same_duration_end_at_rest_and_orbit_keeps_target():
    env = make(scene_id="textured_cube", backend=RectPlane)
    env.reset(seed=0)
    st = env.camera_state()
    from nbm_sim.geometry import project
    lib = make_library()
    assert len(lib) == 18
    for prim in lib:
        sched = instantiate(prim, st, env.cfg)
        assert sum(d for _, d in sched) == pytest.approx(1.0)
        pred = predict_path(env, sched)
        assert pred["feasible"] and pred["at_rest"], prim.name
        if prim.kind == "orbit":
            uv0 = project(env.K, pred["poses"][0], st["target_center_prior"][None])[0][0]
            uv1 = project(env.K, pred["poses"][-1], st["target_center_prior"][None])[0][0]
            assert np.linalg.norm(uv1 - uv0) < 3.0, prim.name
        if prim.kind == "rotate":
            assert pred["path_length"] == pytest.approx(0.0, abs=1e-12)


def test_infeasible_primitive_detected():
    env = make()
    env.reset(seed=0)
    sched = [(np.array([0, 0, 0.2, 0, 0, 0]), 0.05)] * 80                # drive into the plane / workspace edge
    pred = predict_path(env, sched)
    assert not pred["feasible"] and pred["reason"] in ("workspace_violation", "collision_attempt")


# ---------- observability score ----------
def test_score_prefers_motion_across_edges():
    """Vertical edge: horizontal motion fires it, vertical motion slides along it."""
    env = make(EdgePlane, max_linear_acceleration=2.0)
    vol = VotingVolume((-0.3, 0.8, 1.0), (0.3, 1.2, 1.4), 0.01)
    vol.update(env.reset(seed=0))
    for _ in range(6):
        vol.update(env.step(np.array([0.2, 0, 0, 0, 0, 0])))
    for _ in range(6):
        vol.update(env.step(np.zeros(6)))
    a = vol.analyze()
    assert float(a["linearity"].median()) > 0.5
    lib = {p.name: p for p in make_library()}
    rows = {r[2]["name"]: r[2] for r in ObservabilityPlanner([lib["slide_0"], lib["slide_90"]]).score_all(env, vol)}
    assert rows["slide_0"]["score"] > 5 * rows["slide_90"]["score"]


def test_planner_choose_returns_best_feasible():
    env = make()
    vol = VotingVolume(*box(), 0.02)
    vol.update(env.reset(seed=0))
    run(env, vol, ["slide_0"])
    prim, sched, rows = ObservabilityPlanner(make_library()).choose(env, vol)
    best = max(r["score"] for r in rows)
    assert next(r for r in rows if r["name"] == prim.name)["score"] == best and len(sched) == 20


# ---------- oracle branching ----------
def test_branch_restores_state_and_does_not_record(tmp_path):
    env = make(record=True, output_root=str(tmp_path))
    vol, log = VotingVolume(*box(), 0.02), StreamLog()
    p = env.reset(seed=1)
    vol.update(p)
    log.add(p)
    run(env, vol, ["slide_0"], log=log)
    lib = {x.name: x for x in make_library()}
    planner = ObservabilityPlanner([lib["slide_90"], lib["orbit_0"]])
    before = env.get_state()
    rows_written = env.recorder.f["steps/step_index"].shape[0]
    votes = vol.votes.clone()
    base, rows = branch_gains(env, vol, planner, lambda v, lg: float(len(lg.arrays()[0]["t"])), log)
    after = env.get_state()
    assert after["tick"] == before["tick"] and np.array_equal(after["motion"]["T_wc"], before["motion"]["T_wc"])
    assert env.recorder.f["steps/step_index"].shape[0] == rows_written and torch.equal(vol.votes, votes)
    assert all(r["feasible"] and r["gain"] > 0 for r in rows)
    v = instantiate(lib["slide_90"], env.camera_state(), env.cfg)
    real = env.step(*v[0])
    assert real["step_index"] == before["step_index"]


# ---------- GPERT I/O ----------
def test_gpert_export_roundtrip(tmp_path):
    env = make()
    vol, log = VotingVolume(*box(), 0.02), StreamLog()
    p = env.reset(seed=0)
    log.add(p)
    run(env, vol, ["slide_0"], log=log)
    meta = export_gpert(tmp_path, log, origin=(0.0, 1.0, 1.2))
    ev = np.load(tmp_path / "raw_events.npz")
    po = np.load(tmp_path / "camera_poses.npz")
    ca = np.load(tmp_path / "camera_calibration.npz")
    assert ev["position"].shape == (meta["n_events"], 2) and ev["timestamp"].dtype == np.int64
    assert set(np.unique(ev["polarity"])) <= {-1, 1}
    assert np.all(np.diff(po["T_wc_timestamp"]) > 0) and po["T_wc_orientation"].shape[1] == 4
    from scipy.spatial.transform import Rotation
    _, pose_t, pose_T = log.arrays()
    R0 = Rotation.from_quat(po["T_wc_orientation"][0]).as_matrix()          # xyzw
    assert np.allclose(R0, pose_T[0, :3, :3]) and np.allclose(po["T_wc_position"][0], pose_T[0, :3, 3] - [0, 1, 1.2])
    assert int(ca["img_width"]) == 160 and np.allclose(ca["intrinsics"], env.K)
    assert meta["removed_static_intervals"]                                  # the primitive ends at rest
    assert ev["timestamp"].max() <= po["T_wc_timestamp"].max()


def test_compress_static_keeps_trajectory_continuous():
    t = np.arange(10) * 0.01
    T = np.repeat(np.eye(4)[None], 10, 0)
    T[:3, 0, 3] = [0.0, 0.1, 0.2]
    T[3:7, 0, 3] = 0.2                                                       # static from t=0.02 to t=0.06
    T[7:, 0, 3] = [0.3, 0.4, 0.5]
    nt, keep, ev_t, removed = compress_static(t, T, np.array([0.015, 0.04, 0.07]), min_gap=0.01)
    assert removed == [(0.02, 0.06)]
    assert np.allclose(nt, [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]) and list(keep) == [0, 1, 2, 7, 8, 9]
    assert np.allclose(ev_t, [0.015, 0.02, 0.03])


def test_gpert_config_and_ply_reader(tmp_path):
    cfg = gpert_config(tmp_path / "data", tmp_path / "out", 0.2, n_events=100000, duration=2.0)
    for key in ("diff_method", "dataloader_method", "data_type", "interp_method", "accumulation_method",
                "background_color", "log_method"):
        assert key in cfg
    assert cfg["c"] == 0.2 and cfg["is_color"] is False and cfg["accumulation_num"] == 2000
    assert json.loads(json.dumps(cfg)) == cfg
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
             "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros(3, dtype=[(p, "<f4") for p in props])
    data["x"], data["opacity"] = [0.1, 0.2, 0.3], [5.0, -5.0, 0.0]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 3\n" + "".join(
        f"property float {p}\n" for p in props) + "end_header\n"
    (tmp_path / "g.ply").write_bytes(header.encode() + data.tobytes())
    xyz, alpha = load_gpert_points(tmp_path / "g.ply", 0.5, origin=(0, 1, 0))
    assert np.allclose(xyz, [[0.1, 1, 0], [0.3, 1, 0]]) and np.allclose(alpha, [1 / (1 + np.exp(-5)), 0.5])


# ---------- scene texture ----------
def test_edge_texture():
    a, b = make_texture(7, style="edges"), make_texture(7, style="edges")
    assert np.array_equal(a, b) and a.dtype == np.uint8
    assert (np.abs(np.diff(a[..., 0].astype(int), axis=1)) > 20).mean() > 0.005
    with pytest.raises(ValueError):
        SimConfig(texture_style="stripes").validate()
