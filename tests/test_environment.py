import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("dvs_gen")

from nbm_sim.baselines import FixedOrbit, bootstrap_lateral_scan, run_planner
from nbm_sim.packet import concat_events, validate_packet
from nbm_sim.recording import compare_event_streams, iter_packets, load_episode, rerun


def all_events(pkts):
    return concat_events([p["events"] for p in pkts])


def test_reset_and_step_packets_are_valid(make_env):
    env = make_env()
    p0 = env.reset(seed=0)
    assert validate_packet(p0, 64, 48) == [] and p0["rgb"].shape == (1, 48, 64, 3) and len(p0["events"]["t"]) == 0
    pkts = env.run_bootstrap(bootstrap_lateral_scan(env.cfg))
    for p in pkts:
        assert validate_packet(p, 64, 48) == [], validate_packet(p, 64, 48)
        assert p["rgb"].shape[0] == 1 and np.isclose(p["rgb_t"][0], p["t_end"])
        assert p["pose_T_wc"].shape == (51, 4, 4)
    assert sum(len(p["events"]["t"]) for p in pkts) > 0
    assert np.linalg.norm(env.motion.v_w) < 1e-12        # bootstrap ends at rest
    for a, b in zip(pkts[:-1], pkts[1:]):
        assert a["t_end"] == b["t_start"] and np.array_equal(a["T_wc_end"], b["T_wc_start"])


def test_packets_are_immutable(make_env):
    env = make_env()
    p = env.reset()
    with pytest.raises(ValueError):
        p["pose_T_wc"][0, 0, 0] = 5.0


def test_stationary_interval_gives_empty_events(make_env):
    env = make_env()
    env.reset()
    p = env.step(np.zeros(6))
    assert len(p["events"]["t"]) == 0 and p["events"]["x"].dtype == np.uint16
    assert validate_packet(p, 64, 48) == []


def test_malformed_commands_do_not_advance_time(make_env):
    env = make_env()
    env.reset()
    for bad in ([0, 0, 0], [np.nan] * 6, [[0] * 6]):
        with pytest.raises(ValueError):
            env.step(np.array(bad, dtype=float))
    with pytest.raises(ValueError):
        env.step(np.zeros(6), duration=0.0505)
    assert env.tick == 0


def test_continuous_and_split_schedules_are_identical(make_env):
    """Reference persistence: sensor state survives packet boundaries."""
    env = make_env()
    v = np.array([0.1, 0.02, 0.0, 0.0, 0.05, 0.1])
    env.reset(seed=3)
    one = env.step(v, duration=1.0)
    env.reset(seed=3)
    split = [env.step(v, duration=0.05) for _ in range(20)]
    a, b = one["events"], all_events(split)
    assert len(a["t"]) > 0 and compare_event_streams(a, b)["identical"]
    assert np.abs(one["T_wc_end"] - split[-1]["T_wc_end"]).max() < 1e-12


def test_truncation_after_max_steps(make_env):
    env = make_env(max_steps=3)
    env.reset()
    ps = [env.step(np.zeros(6)) for _ in range(3)]
    assert ps[-1]["truncated"] and ps[-1]["reason"] == "budget_exhausted" and not ps[-2]["truncated"]
    with pytest.raises(RuntimeError):
        env.step(np.zeros(6))


def test_safety_termination_reported(make_env):
    env = make_env(max_linear_acceleration=100.0)
    env.reset()
    p = env.step(np.array([0, 0, 0.2, 0, 0, 0]), duration=5.0)   # forward toward the plane/workspace edge
    assert p["terminated"] and p["safety_intervention"] and p["reason"] in ("workspace_violation", "collision_attempt")
    assert p["t_end"] < p["t_start"] + 5.0


def test_observation_isolation(make_env):
    env = make_env(observation_protocol="events_only")
    p = env.reset()
    assert "rgb" not in p and "intensity" not in p
    for forbidden in ("depth", "depth_gt", "mask", "target_mask", "mesh", "depth_valid"):
        assert forbidden not in p
    env2 = make_env()
    p2 = env2.reset()
    assert "rgb" in p2 and "intensity" not in p2 and "depth_observed" not in p2


def test_snapshot_restore_reproduces(make_env):
    env = make_env()
    env.reset(seed=1)
    env.step(np.array([0.1, 0, 0, 0, 0.1, 0]))
    s = env.get_state()
    a = env.step(np.array([0.05, 0.05, 0, 0, 0, 0.2]))
    env.set_state(s)
    b = env.step(np.array([0.05, 0.05, 0, 0, 0, 0.2]))
    assert compare_event_streams(a["events"], b["events"])["identical"]
    assert np.array_equal(a["T_wc_end"], b["T_wc_end"])


def test_recording_and_rerun(make_env, tmp_path):
    env = make_env()
    env.reset(seed=5)
    env.run_bootstrap(bootstrap_lateral_scan(env.cfg))
    orbit = FixedOrbit(speed=0.1)
    pk = run_planner(env, orbit, n_steps=10)
    run_dir = env.save_episode()
    data = load_episode(run_dir)
    rebuilt = list(iter_packets(data))
    assert len(rebuilt) == 1 + 20 + len(pk)
    for r in rebuilt[1:]:
        e = r["events"]
        assert len(e["t"]) == 0 or (e["t"].min() > r["t_start"] and e["t"].max() <= r["t_end"])
    rec = {k: data["obs"][f"events/{k}"] for k in "xytp"}
    env2 = make_env()
    again = rerun(env2, run_dir)
    assert compare_event_streams(rec, all_events(again))["identical"]
    assert data["eval"]["depth"].shape[0] == 1 + 20 + len(pk)
    assert np.isnan(data["eval"]["depth"]).sum() == (~data["eval"]["depth_valid"]).sum()


def test_orbit_respects_limits(make_env):
    from nbm_sim.evaluation import check_motion_limits
    env = make_env(scene_id="textured_cube", max_steps=60)
    env.reset()
    pk = run_planner(env, FixedOrbit())
    tw = np.concatenate([p["executed_twist"] for p in pk])
    poses = np.concatenate([pk[0]["pose_T_wc"][:1]] + [p["pose_T_wc"][1:] for p in pk])
    assert check_motion_limits(tw, poses, env.dt, env.cfg)["ok"]
    assert not pk[-1]["terminated"]
