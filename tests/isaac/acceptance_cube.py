# Isaac-side acceptance checks on the textured-cube scene (spec §17): scene framing, static
# false-event rate, reference persistence, temporal convergence, motion limits, synchronization,
# schema, replay, and the first fixed-trajectory demo. Run with the Isaac Sim Python:
#   python tests/isaac/acceptance_cube.py      (or ${ISAACLAB}/isaaclab.sh -p ...)

# %% Parameters
HEADLESS = True
OUT_DIR = "./acceptance"
STATIC_SECONDS = 1.0
TEMPORAL_DTS = (0.001, 0.0005, 0.00025)       # render intervals compared; last one is the reference
TEMPORAL_TWIST = (0.08, 0.0, 0.0, 0.0, 0.15, 0.0)
TEMPORAL_SECONDS = 0.2

# %% Launch Isaac and build the scene
import json
import pathlib

import numpy as np

from nbm_sim.camera import launch_app
from nbm_sim.config import SimConfig

cfg = SimConfig(scene_id="textured_cube", display=not HEADLESS, record=True, output_root=f"{OUT_DIR}/runs")
app = launch_app(cfg)

from nbm_sim.baselines import FixedOrbit, bootstrap_lateral_scan, run_planner
from nbm_sim.environment import CameraNBMEnv
from nbm_sim.evaluation import (check_motion_limits, count_discrepancy, static_false_event_rate,
                                timing_discrepancy)
from nbm_sim.packet import concat_events, validate_packet
from nbm_sim.recording import compare_event_streams, load_episode, rerun

env = CameraNBMEnv(cfg)
W, H = cfg.width, cfg.height
report = dict(scene="textured_cube", describe=env.describe())


def all_events(pkts):
    return concat_events([p["events"] for p in pkts])


def poses_of(pkts):
    return np.concatenate([pkts[0]["pose_T_wc"][:1]] + [p["pose_T_wc"][1:] for p in pkts])


# %% Scene load: target framed and start pose collision-free
p0 = env.reset(seed=0)
mask = env.get_evaluation_state()["latest"]["mask"]
border = mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
report["scene_load"] = dict(target_pixel_fraction=float(mask.mean()), touches_border=bool(border),
                            initial_T_wc=env.motion.T_wc.tolist(), passed=bool(mask.mean() > 0.01 and not border))

# %% Static rendered scene: false-event rate after warm-up
n = round(STATIC_SECONDS / cfg.action_dt)
static = [env.step(np.zeros(6)) for _ in range(n)]
r = static_false_event_rate(all_events(static), W, H, STATIC_SECONDS)
report["static_scene"] = dict(**r, gate=0.01, passed=bool(r["rate"] <= 0.01))
env.save_episode()

# %% Reference persistence: one continuous schedule vs the same schedule split into packets
v = np.array([0.1, 0.0, 0.0, 0.0, 0.1, 0.0])
env.reset(seed=0)
one = env.step(v, duration=1.0)
env.save_episode()
env.reset(seed=0)
split = [env.step(v) for _ in range(20)]
env.save_episode()
cmp = compare_event_streams(one["events"], all_events(split))
dT = float(np.abs(one["T_wc_end"] - split[-1]["T_wc_end"]).max())
report["reference_persistence"] = dict(**cmp, max_pose_diff=dT, passed=bool(cmp["identical"] and dT < 1e-12),
                                       note="exact equality requires a deterministic renderer")

# %% Temporal sampling convergence (acceleration limits off so poses match at common times)
streams = {}
for dt in TEMPORAL_DTS:
    env.set_timing(base_dt=dt, event_render_dt=dt, max_linear_acceleration=float("inf"),
                   max_angular_acceleration=float("inf"))
    env.reset(seed=0)
    streams[dt] = env.step(np.array(TEMPORAL_TWIST), duration=TEMPORAL_SECONDS)["events"]
    env.save_episode()
env.set_timing(base_dt=cfg.base_dt, event_render_dt=cfg.event_render_dt,
               max_linear_acceleration=cfg.max_linear_acceleration, max_angular_acceleration=cfg.max_angular_acceleration)
ref = streams[TEMPORAL_DTS[-1]]
temporal = {}
for dt in TEMPORAL_DTS[:-1]:
    d = count_discrepancy(streams[dt], ref, W, H)
    d.update(timing_discrepancy(streams[dt], ref, W, H))
    temporal[str(dt)] = d
closest = temporal[str(TEMPORAL_DTS[-2])]
report["temporal_sampling"] = dict(
    comparisons=temporal, nonempty=bool(len(ref["t"]) > 0),
    passed=bool(len(ref["t"]) > 0 and max(closest["count_change_pos"], closest["count_change_neg"],
                                          closest["l1_pos"], closest["l1_neg"]) < 0.01),
    note="EVIS emits at most one event per pixel per sample, so counts depend on sampling rate until converged")

# %% Bootstrap + fixed orbit: limits, synchronization, schema, and the first demo recording
env.reset(seed=0)
boot = env.run_bootstrap(bootstrap_lateral_scan(cfg))
orbit = run_planner(env, FixedOrbit(), n_steps=60)
pk = boot + orbit
lim = check_motion_limits(np.concatenate([p["executed_twist"] for p in pk]), poses_of(pk), cfg.base_dt, cfg)
schema = [e for p in pk for e in validate_packet(p, W, H)]
sync = all(np.allclose(p["rgb_T_wc"][0], p["T_wc_end"]) and np.isclose(p["rgb_t"][0], p["t_end"]) for p in pk)
dup = int(sum(len(np.intersect1d(a["events"]["t"], b["events"]["t"])) for a, b in zip(pk[:-1], pk[1:])))
demo_dir = env.save_episode()
report["limits"] = dict(**lim, passed=lim["ok"])
report["synchronization"] = dict(rgb_pose_time_match=bool(sync), passed=bool(sync))
report["schema"] = dict(errors=schema[:20], duplicate_boundary_timestamps=dup, passed=bool(not schema and dup == 0))
d = load_episode(demo_dir)
report["first_demo"] = dict(run_dir=demo_dir, n_events=int(len(d["obs"].get("events/t", []))),
                            n_rgb=int(len(d["obs"]["rgb_t"])), n_depth=int(len(d["eval"]["depth_t"])),
                            passed=bool(len(d["obs"].get("events/t", [])) > 0 and len(d["eval"]["depth_t"]) > 0))

# %% Replay: rerun the recorded command schedule from its reset state
again = rerun(env, demo_dir)
env.save_episode()
rec = {k: d["obs"][f"events/{k}"] for k in "xytp"}
cmp = compare_event_streams(rec, all_events(again))
dpose = float(np.abs(d["traj"]["pose_T_wc"] - poses_of(again)).max())
report["replay"] = dict(**cmp, max_pose_diff=dpose, passed=bool(cmp["identical"] and dpose < 1e-12))

# %% Write report
report["all_passed"] = all(v["passed"] for v in report.values() if isinstance(v, dict) and "passed" in v)
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT_DIR, "acceptance_cube.json").write_text(json.dumps(report, indent=2, default=str))
print(json.dumps({k: v.get("passed") for k, v in report.items() if isinstance(v, dict) and "passed" in v}, indent=2))
env.close()
app.close()
