import numpy as np

from nbm_sim.packet import PROTOCOL_FIELDS, actor_view, concat_events, empty_events


def test_empty_events_shapes_and_dtypes():
    e = empty_events()
    assert e["x"].dtype == np.uint16 and e["t"].dtype == np.float64 and e["p"].dtype == np.int8
    assert all(v.shape == (0,) for v in e.values())


def test_concat_sorts_stably():
    a = dict(x=np.array([5, 6]), y=np.array([0, 0]), t=np.array([0.2, 0.2]), p=np.array([1, -1]))
    b = dict(x=np.array([7]), y=np.array([0]), t=np.array([0.1]), p=np.array([1]))
    e = concat_events([a, b])
    assert list(e["x"]) == [7, 5, 6]


def test_protocol_fields_never_include_privileged():
    privileged = {"depth", "depth_gt", "depth_valid", "target_mask", "mask", "mesh", "normals", "error"}
    for fields in PROTOCOL_FIELDS.values():
        assert not fields & privileged
    p = {k: 0 for k in PROTOCOL_FIELDS["oracle"] | {"depth_gt"}}
    assert "depth_gt" not in actor_view(p, "oracle")
    assert "intensity" not in actor_view(p, "events_only", expose_intensity=True)
