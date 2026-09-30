import pytest

from nbm_sim import camera
from nbm_sim.config import SimConfig


@pytest.mark.parametrize("lab,order,hdr", [("2.3.2", "wxyz", "HdrColor"), ("3.0.0", "xyzw", "rgb_hdr")])
def test_supported_lines(monkeypatch, lab, order, hdr):
    monkeypatch.setattr(camera, "_version", lambda p: lab if p == "isaaclab" else None)
    c = camera.IsaacCompat(SimConfig())
    assert c.quat_order == order and c.hdr_channel == hdr


def test_unsupported_line_refused(monkeypatch):
    monkeypatch.setattr(camera, "_version", lambda p: "2.1.0" if p == "isaaclab" else None)
    with pytest.raises(RuntimeError):
        camera.IsaacCompat(SimConfig())
    c = camera.IsaacCompat(SimConfig(allow_untested_isaac=True))
    assert c.quat_order == "wxyz"


def test_explicit_quat_order_override(monkeypatch):
    monkeypatch.setattr(camera, "_version", lambda p: "3.0.0" if p == "isaaclab" else None)
    assert camera.IsaacCompat(SimConfig(isaac_quat_order="wxyz")).quat_order == "wxyz"
