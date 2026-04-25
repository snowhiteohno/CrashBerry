"""Environment wrapper for the Incident Response Simulator.
Provides the OpenAI‑Gym compatible API used by agents, evaluation, and training.
Implements reset(), step(), render(), and seed() delegating logic to Simulator.
"""
import json
from typing import Any, Dict, Tuple

from .simulator import Simulator

class IncidentResponseEnv:
    """Gym‑style environment exposing the PRD specifications.
    
    The observation dict matches the format described in the PRD and is
    returned by ``Simulator._generate_observation``. ``step`` simply forwards
    the action to the underlying ``Simulator`` and returns the tuple
    ``(observation, reward, done, info)``.
    """

    def __init__(self, seed: int | None = None):
        self.seed(seed)
        self._sim = Simulator(seed=self._seed)
        self.max_steps = 20

    # ---------------------------------------------------------------------
    # Gym API
    # ---------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> Dict[str, Any]:
        """Reset the environment and return the first observation.
        Allows an optional seed to create reproducible episodes.
        """
        if seed is not None:
            self.seed(seed)
        self._sim = Simulator(seed=self._seed)
        return self._sim.reset()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply ``action`` and return ``(observation, reward, done, info)``.
        ``action`` follows the PRD schema, e.g. ``{"type": "diagnose", "target": "database", "failure_mode": "memory_leak"}``.
        """
        return self._sim.step(action)

    def render(self, mode: str = "json") -> str:
        """Return a human‑readable representation of the current hidden state.
        In ``json`` mode we dump the internal true health values; other modes
        could be extended later.
        """
        if mode == "json":
            return json.dumps({"true_health": self._sim.true_health}, indent=2)
        return str(self._sim.true_health)

    def seed(self, seed: int | None = None) -> None:
        """Set the random seed for reproducibility.
        The seed is stored and passed to the underlying ``Simulator`` on reset.
        """
        self._seed = seed if seed is not None else 0
        # ``random`` module in ``Simulator`` will be re‑initialised on next reset.

    # ---------------------------------------------------------------------
    # Convenience helpers used by agents/evaluation
    # ---------------------------------------------------------------------
    @property
    def action_space(self):
        """Return a list of allowed action types – helpful for random agents."""
        return ["diagnose", "restart_service", "rollback_deploy", "scale_up", "enable_circuit_breaker", "no_op", "check_logs"]

    @property
    def observation_space(self):
        """Placeholder – agents rely on the dictionary format rather than a fixed gym space."""
        return None

    def close(self):
        """Placeholder for compatibility with Gym environments."""
        pass
