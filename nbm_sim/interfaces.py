"""Interchangeable research-module interfaces (spec §12). The simulator depends on none of them."""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class Reconstructor(Protocol):
    def update(self, packet: dict) -> Any:
        """Incremental update from one actor-view packet with known poses; returns ``map_state``."""


class UncertaintyModel(Protocol):
    def update(self, map_state: Any, observation_history: list) -> dict:
        """Must declare: frame, representation, resolution, support/validity, definition, units, and keep
        unobserved space, unreliable observed geometry and predicted measurement uncertainty distinct."""


class Planner(Protocol):
    def act(self, map_state: Any, uncertainty: dict, observation_history: list, camera_state: dict) -> np.ndarray:
        """Return a physical body twist ``[vx, vy, vz, wx, wy, wz]`` (m/s, rad/s)."""


class EventPredictor(Protocol):
    def predict(self, map_state: Any, current_observation: dict, camera_state: dict, candidate_velocity: np.ndarray,
                duration: float) -> dict:
        """Return predicted events or count maps plus support/uncertainty, using only past/current
        permitted information. Unknown map regions stay unknown (never zero predicted events)."""
