# Option C: fast ray-voting map in the loop, primitive library as the action set, GPERT offline for the final score.
# Run with the Isaac Sim Python:  python experiment_option_c.py   (or ${ISAACLAB}/isaaclab.sh -p experiment_option_c.py)

# %% Parameters
SCENE_ID = "textured_cube"
SEED = 0
HEADLESS = True
START = "slide_0"            # start primitive, a name from make_library()
N_DECISIONS = 9              # primitives last 1 s; horizon = N_DECISIONS s after the start
VOXEL = 0.002                # m
DEVICE = "cuda:0"
OUTPUT_ROOT = "./runs"
EXPORT_GPERT = True

# %% Launch Isaac and create the environment
import json

import numpy as np

from nbm_sim.camera import launch_app
from nbm_sim.config import SimConfig

cfg = SimConfig(scene_id=SCENE_ID, seed=SEED, display=not HEADLESS, output_root=OUTPUT_ROOT, device=DEVICE,
                observation_protocol="events_only", texture_style="edges", horizon_seconds=float(N_DECISIONS))
app = launch_app(cfg)

from nbm_sim.environment import CameraNBMEnv
from nbm_sim.gpert_io import StreamLog, export_gpert, gpert_config, write_config
from nbm_sim.planner import ObservabilityPlanner, oracle_fscore
from nbm_sim.primitives import instantiate, make_library
from nbm_sim.voting import VotingVolume

env = CameraNBMEnv(cfg)
library = make_library()
planner = ObservabilityPlanner(library)
vol = VotingVolume.for_scene(env.spec, VOXEL, device=DEVICE)
log = StreamLog()
score = oracle_fscore(env)          # evaluator-only, for the history log; never seen by the planner

# %% Start primitive (bootstrap)
p = env.reset(seed=SEED)
vol.update(p)
log.add(p)
for v, d in instantiate(next(x for x in library if x.name == START), env.camera_state(), cfg):
    p = env.step(v, d, phase="bootstrap")
    vol.update(p)
    log.add(p)
history = [dict(decision="start", primitive=START, fscore=score(vol), **vol.summary())]

# %% Plan loop
for k in range(N_DECISIONS):
    choice = planner.choose(env, vol)
    if choice is None:
        break
    prim, sched, rows = choice
    for v, d in sched:
        p = env.step(v, d)
        vol.update(p)
        log.add(p)
        if p["terminated"] or p["truncated"]:
            break
    history.append(dict(decision=k, primitive=prim.name, scores=rows, fscore=score(vol), **vol.summary()))
    if p["terminated"] or p["truncated"]:
        break

# %% Save the episode and the GPERT input for the final score
run_dir = env.save_episode()
json.dump(history, open(f"{run_dir}/option_c_history.json", "w"), indent=2, default=float)
if EXPORT_GPERT:
    lo, hi = (np.asarray(b) for b in env.spec.task_region(0.05))
    meta = export_gpert(f"{run_dir}/gpert/data", log, (lo + hi) / 2)
    write_config(f"{run_dir}/gpert/config.yaml",
                 gpert_config(f"{run_dir}/gpert/data", f"{run_dir}/gpert/out", float((hi - lo).max() / 2),
                              n_events=meta["n_events"], duration=meta["duration"]))
print(run_dir, [(h["primitive"], round(h["fscore"], 3)) for h in history])

# %%
env.close()
app.close()
