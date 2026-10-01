"""Regression tests for the 2026-10-01 review: episode lifecycle, time budget, accelerated scheduling, RGB-D."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("dvs_gen")

from nbm_sim.config import SimConfig
from nbm_sim.packet import validate_packet
from nbm_sim.recording import iter_packets, load_episode, rerun, compare_event_streams
from nbm_sim.packet import concat_events


def inside(p):
    t = p["events"]["t"]
    return len(t) == 0 or (t.min() > p["t_start"] and t.max() <= p["t_end"])


# 1. lifecycle
def test_set_timing_after_save_episode(make_env):
    env = make_env()
    env.reset()
    env.step(np.zeros(6))
    env.save_episode()
    env.set_timing(base_dt=0.0005, event_render_dt=0.0005)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(6))
    env.reset()
    assert env.step(np.zeros(6))["pose_T_wc"].shape == (101, 4, 4)


def test_end_episode_without_record(make_env):
    env = make_env(record=False)
    env.reset()
    env.step(np.zeros(6))
    with pytest.raises(RuntimeError):
        env.set_timing(base_dt=0.0005, event_render_dt=0.0005)
    env.end_episode()
    env.set_timing(base_dt=0.0005, event_render_dt=0.0005)


# 2. time budget
def test_long_action_stops_at_remaining_budget(make_env):
    env = make_env(record=False, horizon_seconds=0.1)
    env.reset()
    p = env.step(np.zeros(6), duration=0.3)
    assert np.isclose(p["t_end"] - p["t_start"], 0.1) and p["truncated"] and p["diagnostics"]["budget_clipped"]
    assert np.isclose(p["diagnostics"]["requested_duration"], 0.3)


def test_budget_excludes_bootstrap(make_env):
    env = make_env(record=False, horizon_seconds=0.1)
    env.reset()
    env.run_bootstrap([(np.zeros(6), 0.05)] * 4)
    p = env.step(np.zeros(6), duration=0.3)
    assert np.isclose(p["t_start"], 0.2) and np.isclose(p["t_end"], 0.3) and p["truncated"]


def test_bootstrap_after_policy_rejected(make_env):
    env = make_env(record=False)
    env.reset()
    env.step(np.zeros(6))
    with pytest.raises(RuntimeError):
        env.step(np.zeros(6), phase="bootstrap")


# 3. accelerated mode
def test_accelerated_defaults_run_and_keep_events_in_packets(make_env):
    env = make_env(record=False, event_mode="accelerated", max_linear_acceleration=100.0)
    env.reset()
    pk = env.run_bootstrap([(np.array([0.2, 0, 0, 0, 0, 0]), 0.05)] * 3)
    pk += [env.step(np.array([0.2, 0, 0, 0, 0, 0])) for _ in range(3)]
    assert sum(len(p["events"]["t"]) for p in pk) > 0
    assert all(inside(p) for p in pk)
    assert all(validate_packet(p, 64, 48) == [] for p in pk)


def test_accelerated_incompatible_settings_rejected_upfront():
    with pytest.raises(ValueError):
        SimConfig(event_mode="accelerated", keyframe_dt=0.008).validate()
    with pytest.raises(ValueError):
        SimConfig(event_mode="accelerated", keyframe_dt=0.010, horizon_seconds=10.005).validate()


def test_accelerated_misaligned_duration_rejected_without_advancing(make_env):
    env = make_env(record=False, event_mode="accelerated")
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.zeros(6), duration=0.015)
    with pytest.raises(ValueError):
        env.step(np.zeros(6), duration=0.015, phase="bootstrap")
    assert env.tick == 0


def test_accelerated_emergency_stop_closes_gap(make_env):
    env = make_env(record=False, event_mode="accelerated", max_linear_acceleration=100.0)
    env.reset()
    p = env.step(np.array([0, 0, 0.2, 0, 0, 0]), duration=5.0)
    assert p["terminated"] and env.tick % env.n_key != 0
    assert env._last_key_tick == env.tick and inside(p)


def test_accelerated_snapshot_restore(make_env):
    env = make_env(record=False, event_mode="accelerated", max_linear_acceleration=100.0)
    env.reset(seed=2)
    env.step(np.array([0.1, 0, 0, 0, 0, 0]))
    s = env.get_state()
    a = env.step(np.array([0.1, 0.05, 0, 0, 0, 0.1]))
    env.set_state(s)
    b = env.step(np.array([0.1, 0.05, 0, 0, 0, 0.1]))
    assert compare_event_streams(a["events"], b["events"])["identical"]


# 4. RGB-D protocol
def test_rgbd_packets_carry_depth_observed(make_env):
    env = make_env(record=False, observation_protocol="events_rgbd_known_pose", depth_observation_source="simulator")
    p0 = env.reset()
    p = env.step(np.zeros(6))
    for q in (p0, p):
        assert q["depth_observed"].shape == (1, 48, 64) and q["depth_observed"].dtype == np.float32
        assert validate_packet(q, 64, 48) == []
    assert np.isclose(p["depth_observed_t"][0], p["t_end"]) and np.allclose(p["depth_observed_T_wc"][0], p["T_wc_end"])
    z = p["depth_observed"][0]
    assert np.isfinite(z).any() and np.nanmax(np.abs(z - 1.0)) < 1e-5   # exact GT plane at Z = 1 m


def test_rgbd_noise_and_dropout_are_seeded(make_env):
    kw = dict(record=False, observation_protocol="events_rgbd_known_pose", depth_observation_source="simulator",
              depth_noise_std_at_1m=0.01, depth_dropout=0.2)
    a = make_env(**kw).reset(seed=4)["depth_observed"]
    b = make_env(**kw).reset(seed=4)["depth_observed"]
    c = make_env(**kw).reset(seed=5)["depth_observed"]
    assert np.array_equal(a, b, equal_nan=True) and not np.array_equal(a, c, equal_nan=True)
    assert 0.1 < np.isnan(a).mean() < 0.3 and 0.001 < np.nanstd(a - 1.0) < 0.05


def test_depth_not_exposed_to_other_protocols(make_env):
    with pytest.raises(ValueError):
        SimConfig(observation_protocol="events_rgb_known_pose", depth_observation_source="simulator").validate()
    with pytest.raises(ValueError):
        SimConfig(observation_protocol="events_rgbd_known_pose").validate()
    assert "depth_observed" not in make_env(record=False).reset()


def test_rgbd_recording_roundtrip(make_env):
    env = make_env(observation_protocol="events_rgbd_known_pose", depth_observation_source="simulator",
                   depth_noise_std_at_1m=0.01, horizon_seconds=0.1)
    env.reset(seed=1)
    live = [env.step(np.array([0.05, 0, 0, 0, 0, 0]), duration=0.3)]
    run = env.save_episode()
    rec = list(iter_packets(load_episode(run)))
    assert np.array_equal(rec[1]["depth_observed"], live[0]["depth_observed"], equal_nan=True)
    again = rerun(make_env(observation_protocol="events_rgbd_known_pose", depth_observation_source="simulator",
                           depth_noise_std_at_1m=0.01, horizon_seconds=0.1), run)
    assert np.array_equal(again[1]["depth_observed"], live[0]["depth_observed"], equal_nan=True)
    assert compare_event_streams(concat_events([p["events"] for p in again]), live[0]["events"])["identical"]
