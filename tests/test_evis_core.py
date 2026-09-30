"""Probes of the EVIS core behavior through its own API (no second generator is written).

These record what the package does; the NBM spec's expectations are checked against them."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("dvs_gen")

from nbm_sim.config import SimConfig
from nbm_sim.events import EvisEventCamera, package_info


def cam(**kw):
    return EvisEventCamera(SimConfig(**kw))


def frame(val, H=4, W=5):
    return torch.full((1, H, W, 3), float(val))


def test_first_frame_initializes_without_events():
    c = cam()
    c.process(frame(0.5), 0.0)
    assert len(c.drain()["t"]) == 0


def test_constant_input_gives_zero_events():
    c = cam()
    for i in range(100):
        c.process(frame(0.3), i * 1e-3)
    assert len(c.drain()["t"]) == 0


def test_brightness_ramp_polarity_and_threshold():
    c = cam(event_threshold=0.2)
    c.process(frame(0.1), 0.0)
    c.process(frame(0.1 * np.exp(0.19)), 1e-3)
    assert len(c.drain()["t"]) == 0
    c.process(frame(0.1 * np.exp(0.21)), 2e-3)
    e = c.drain()
    assert len(e["t"]) == 20 and np.all(e["p"] == 1) and np.all(e["t"] == 2e-3)
    c.process(frame(0.1 * np.exp(0.21 - 0.25)), 3e-3)
    e = c.drain()
    assert len(e["t"]) == 20 and np.all(e["p"] == -1)


def test_large_step_emits_one_event_per_pixel_per_sample():
    """Package behavior: a change of several thresholds in one sample yields ONE event and the
    reference latches to the new value (no multiple crossings)."""
    c = cam(event_threshold=0.2)
    c.process(frame(0.1), 0.0)
    c.process(frame(0.1 * np.exp(1.0)), 1e-3)          # 5 thresholds
    assert len(c.drain()["t"]) == 20
    c.process(frame(0.1 * np.exp(1.0)), 2e-3)
    assert len(c.drain()["t"]) == 0
    ref = c.reference_state()
    assert np.allclose(ref, np.log(0.1 * np.exp(1.0) + 1e-5), atol=1e-5)


def test_reset_reinitializes_reference():
    c = cam()
    c.process(frame(0.1), 0.0)
    c.reset()
    c.process(frame(0.9), 0.1)
    assert len(c.drain()["t"]) == 0


def test_state_roundtrip():
    c = cam()
    c.process(frame(0.1), 0.0)
    s = c.get_state()
    c.process(frame(0.5), 1e-3)
    a = c.drain()
    c.set_state(s)
    c.process(frame(0.5), 1e-3)
    b = c.drain()
    assert all(np.array_equal(a[k], b[k]) for k in "xytp") and len(a["t"]) == 20


def test_package_info_verifies_model_constants():
    info = package_info()
    assert "UNVERIFIED" not in str(info["model_as_found_in_source"]), info
