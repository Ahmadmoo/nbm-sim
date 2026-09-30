import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from nbm_sim.config import SimConfig


@pytest.fixture
def plane_cfg(tmp_path):
    return SimConfig(scene_id="textured_plane", width=64, height=48, warmup_renders=2, display=False, record=True,
                     output_root=str(tmp_path / "runs"), device="cpu")


@pytest.fixture
def make_env(plane_cfg):
    from fake_backend import PlaneBackend
    from nbm_sim.environment import CameraNBMEnv

    def _make(**kw):
        cfg = plane_cfg.replace(**kw) if kw else plane_cfg
        return CameraNBMEnv(cfg, backend=PlaneBackend(cfg))
    return _make
