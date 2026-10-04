# Option C check: does the observability score predict the real gain of each primitive?
# From each start state, every primitive is run for real (oracle branching, evaluator-only data),
# its true gain is measured, and the ranking is compared with the planner's scores.
# Run with the Isaac Sim Python:  python experiment_branch.py

# %% Parameters
SCENE_ID = "textured_cube"
SEEDS = [0]
STARTS = ["slide_0", "slide_90", "orbit_0", "orbit_90", "orbit_45", "approach"]
VOXEL = 0.002
DEVICE = "cuda:0"
EVALUATOR = "volume"         # "volume": F-score of the voting map (fast) | "gpert": retrain GPERT per branch (slow)
GPERT_ROOT = "../gpert"      # clone of github.com/e3ai/gpert
GPERT_PYTHON = "python"      # Python of GPERT's own environment
GPERT_STEPS = 10000
WORKDIR = "./runs/branch"

# %% Launch Isaac and create the environment
import json
import os

import numpy as np

from nbm_sim.camera import launch_app
from nbm_sim.config import SimConfig

cfg = SimConfig(scene_id=SCENE_ID, display=False, record=False, device=DEVICE, observation_protocol="events_only",
                texture_style="edges", horizon_seconds=10.0)
app = launch_app(cfg)

from nbm_sim.environment import CameraNBMEnv
from nbm_sim.gpert_io import StreamLog, gpert_evaluator
from nbm_sim.planner import ObservabilityPlanner, branch_gains, oracle_fscore, rank_agreement
from nbm_sim.primitives import instantiate, make_library
from nbm_sim.voting import VotingVolume

os.makedirs(WORKDIR, exist_ok=True)
env = CameraNBMEnv(cfg)
library = make_library()
byname = {p.name: p for p in library}
planner = ObservabilityPlanner(library)
evaluate = oracle_fscore(env) if EVALUATOR == "volume" else \
    gpert_evaluator(env, WORKDIR, GPERT_ROOT, GPERT_PYTHON, max_steps=GPERT_STEPS)

# %% Branch from each start state
results = []
for seed in SEEDS:
    for start in STARTS:
        vol, log = VotingVolume.for_scene(env.spec, VOXEL, device=DEVICE), StreamLog()
        p = env.reset(seed=seed)
        vol.update(p)
        log.add(p)
        for v, d in instantiate(byname[start], env.camera_state(), cfg):
            p = env.step(v, d, phase="bootstrap")
            vol.update(p)
            log.add(p)
        scores = [r[2] for r in planner.score_all(env, vol)]
        base, gains = branch_gains(env, vol, planner, evaluate, log)
        results.append(dict(seed=seed, start=start, base=base, scores=scores, gains=gains,
                            agreement=rank_agreement(scores, gains)))
        print(start, results[-1]["agreement"])

# %% Summary
json.dump(results, open(f"{WORKDIR}/branch_results_{EVALUATOR}.json", "w"), indent=2, default=float)
print("median rank correlation:", np.nanmedian([r["agreement"]["rho"] for r in results]),
      "| median gain spread:", np.nanmedian([r["agreement"].get("gain_spread", np.nan) for r in results]))

# %%
env.close()
app.close()
