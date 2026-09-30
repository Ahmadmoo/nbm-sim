import numpy as np

from nbm_sim.evaluation import count_discrepancy, reconstruction_metrics, static_false_event_rate
from nbm_sim.packet import empty_events
from nbm_sim.scene import evaluation_geometry, get_scene


def test_perfect_and_empty_reconstruction():
    spec = get_scene("textured_cube")
    g = evaluation_geometry(spec, spacing=0.004)
    region = spec.task_region()
    m = reconstruction_metrics(g["accessible_surface"], g["accessible_surface"], region)
    assert m["fscore@2mm"] == 1.0 and m["accuracy_mean"] == 0.0
    e = reconstruction_metrics(np.zeros((0, 3)), g["accessible_surface"], region)
    assert e["fscore@5mm"] == 0.0 and np.isinf(e["accuracy_mean"])


def test_table_points_do_not_improve_scores():
    spec = get_scene("textured_cube")
    g = evaluation_geometry(spec, spacing=0.004)
    table = np.c_[np.random.default_rng(0).uniform(-0.5, 0.5, (500, 2)), np.full(500, 0.80)]
    m = reconstruction_metrics(table, g["accessible_surface"], spec.task_region())
    assert m["n_points"] == 0 and m["n_points_outside_region"] == 500


def test_accessible_surface_excludes_bottom():
    g = evaluation_geometry(get_scene("textured_cube"), spacing=0.01)
    bottom = np.isclose(g["full_surface"][:, 2], 0.80)
    interior = np.all(np.abs(g["full_surface"][:, :2]) < 0.1 - 1e-9, axis=1)
    assert (bottom & interior).sum() > 0
    a = g["accessible_surface"]
    assert (np.isclose(a[:, 2], 0.80) & np.all(np.abs(a[:, :2]) < 0.1 - 1e-9, axis=1)).sum() == 0


def test_count_discrepancy():
    ev = dict(x=np.array([1, 2], np.uint16), y=np.array([1, 1], np.uint16), t=np.array([0.1, 0.2]),
              p=np.array([1, -1], np.int8))
    d = count_discrepancy(ev, ev, 4, 4)
    assert d["l1_pos"] == 0 and d["l1_neg"] == 0
    d = count_discrepancy(empty_events(), empty_events(), 4, 4)
    assert d["l1_pos"] == 0
    assert static_false_event_rate(ev, 4, 4, 1.0)["rate"] == 2 / 16
