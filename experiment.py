# First deliverable (M3): fixed-trajectory example on the textured cube.
# Run with the Isaac Sim Python:  python experiment.py   (or ${ISAACLAB}/isaaclab.sh -p experiment.py)

# %% Parameters
SCENE_ID = "textured_cube"
SEED = 0
HEADLESS = False
ORBIT_STEPS = 100            # 5 s of planning time at 0.05 s per step
ORBIT_SPEED = 0.15           # m/s
OUTPUT_ROOT = "./runs"
WRITE_PREVIEW = True

# %% Launch Isaac and create the environment
import json

from nbm_sim.camera import launch_app
from nbm_sim.config import SimConfig

cfg = SimConfig(scene_id=SCENE_ID, seed=SEED, display=not HEADLESS, output_root=OUTPUT_ROOT)
app = launch_app(cfg)

from nbm_sim.baselines import FixedOrbit, bootstrap_lateral_scan, run_planner
from nbm_sim.environment import CameraNBMEnv
from nbm_sim.recording import write_preview

env = CameraNBMEnv(cfg)

# %% Reset, shared bootstrap, fixed orbit
packet = env.reset(seed=SEED)
bootstrap = env.run_bootstrap(bootstrap_lateral_scan(cfg))
orbit = run_planner(env, FixedOrbit(speed=ORBIT_SPEED), n_steps=ORBIT_STEPS)

# %% Save and summarize
run_dir = env.save_episode()
if WRITE_PREVIEW:
    write_preview(run_dir)
metrics = json.load(open(f"{run_dir}/metrics.json"))
print(run_dir, metrics["counts"], f"path {metrics['path_length_m']:.3f} m", f"reason {metrics['reason'] or '-'}")

# %%
env.close()
app.close()
