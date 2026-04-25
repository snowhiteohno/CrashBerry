# env/simulator.py
"""Simulator for the incident response environment.
Implements hidden state, failure propagation, noisy observations, and action effects.
The design follows the PRD specifications.
"""
import random
import copy
import math
from typing import Dict, Any, Tuple, List

# Constants from PRD
SERVICES = ["api-gateway", "auth-service", "database", "cache", "worker"]
FAILURE_MODES = ["crashed", "memory_leak", "overloaded", "bad_deploy"]
# Propagation matrix (rows -> source, cols -> target)
PROPAGATION_MATRIX = [
    [1.0, 0.3, 0.0, 0.0, 0.0],  # api-gateway
    [0.4, 1.0, 0.0, 0.0, 0.0],  # auth-service
    [0.5, 0.2, 1.0, 0.3, 0.6],  # database
    [0.2, 0.0, 0.1, 1.0, 0.3],  # cache
    [0.3, 0.0, 0.0, 0.1, 1.0],  # worker
]
NOISE_STD = 0.15  # Gaussian noise std
HEALTH_LAG_STEPS = 1  # health observation lag
MEMORY_LEAK_RECURRENCE_STEPS = 4  # after fix, leak recurs


def _add_noise(value: float) -> float:
    return max(0.0, min(1.0, random.gauss(value, NOISE_STD)))


class Simulator:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> Dict[str, Any]:
        """Initialize hidden state for a new episode.
        Returns the first observation (with lag applied).
        """
        # true health per service (0..1) start at 1.0
        self.true_health = {svc: 1.0 for svc in SERVICES}
        # pick root cause service and failure mode
        self.root_service = self.rng.choice(SERVICES)
        self.root_failure = self.rng.choice(FAILURE_MODES)
        # apply initial failure effect directly to root service
        self._apply_failure(self.root_service, self.root_failure)
        # step counters
        self.current_step = 0
        self.max_steps = 20
        # lag buffer for health observations
        self.health_history: List[Dict[str, float]] = []
        # track whether diagnosis has been made
        self.diagnosis_made = False
        # memory leak recurrence tracker (steps left until recurrence)
        self.memory_leak_counter = None
        # generate first observation (lag is 0 because no prior step)
        observation = self._generate_observation()
        self.health_history.append(copy.deepcopy(self.true_health))
        return observation

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _apply_failure(self, service: str, mode: str):
        """Mutate true health according to failure mode.
        Effects are defined in the PRD action‑effect table.
        """
        if mode == "crashed":
            self.true_health[service] = 0.0
        elif mode == "memory_leak":
            self.true_health[service] = max(0.0, self.true_health[service] - 0.2)
            # schedule recurrence if the agent later restarts the service
            self.memory_leak_counter = None
        elif mode == "overloaded":
            self.true_health[service] = max(0.0, self.true_health[service] - 0.15)
        elif mode == "bad_deploy":
            # degrade more aggressively
            self.true_health[service] = max(0.0, self.true_health[service] - 0.25)
        # propagate degradation to dependent services using matrix
        src_idx = SERVICES.index(service)
        for tgt_idx, svc in enumerate(SERVICES):
            if tgt_idx == src_idx:
                continue
            factor = PROPAGATION_MATRIX[src_idx][tgt_idx]
            if factor > 0:
                degradation = (1.0 - self.true_health[service]) * factor
                self.true_health[svc] = max(0.0, self.true_health[svc] - degradation)

    def _propagate(self):
        """Re‑apply propagation matrix each step to reflect lingering effects.
        This simple implementation re‑applies the matrix based on current true health.
        """
        new_health = copy.deepcopy(self.true_health)
        for src_idx, src_svc in enumerate(SERVICES):
            src_health = self.true_health[src_svc]
            for tgt_idx, tgt_svc in enumerate(SERVICES):
                if src_idx == tgt_idx:
                    continue
                factor = PROPAGATION_MATRIX[src_idx][tgt_idx]
                if factor > 0:
                    degradation = (1.0 - src_health) * factor
                    new_health[tgt_svc] = max(0.0, new_health[tgt_svc] - degradation)
        self.true_health = new_health

    def _generate_observation(self) -> Dict[str, Any]:
        """Create the observation dict returned to the agent.
        Includes noisy metrics, metric trends, recent alerts, and other fields.
        """
        # Compute lagged health for observation (use history buffer)
        if len(self.health_history) < HEALTH_LAG_STEPS:
            lagged = {svc: 1.0 for svc in SERVICES}
        else:
            lagged = self.health_history[-HEALTH_LAG_STEPS]
        # noisy metrics
        metrics = {}
        for svc in SERVICES:
            metrics[svc] = {
                "cpu": _add_noise(lagged[svc]),
                "error_rate": _add_noise(1.0 - lagged[svc]),
                "latency_ms": int(_add_noise(lagged[svc]) * 2000),
                "queue_depth": int(_add_noise(lagged[svc]) * 200),
            }
        # simplistic trend detection based on previous observation
        trend = {}
        if self.health_history:
            prev = self.health_history[-1]
            for svc in SERVICES:
                diff = lagged[svc] - prev[svc]
                if diff < -0.05:
                    trend[svc] = "degrading"
                elif diff > 0.05:
                    trend[svc] = "recovering"
                else:
                    trend[svc] = "stable"
        else:
            trend = {svc: "stable" for svc in SERVICES}
        # recent alerts (very simple heuristic)
        alerts = []
        for svc in SERVICES:
            if lagged[svc] < 0.5:
                alerts.append(f"{svc}: health low ({lagged[svc]:.2f})")
        # system health score is average of true health (not noisy)
        system_health_score = sum(self.true_health.values()) / len(SERVICES)
        observation = {
            "step": self.current_step,
            "max_steps": self.max_steps,
            "system_health_score": system_health_score,
            "metrics": metrics,
            "metric_trend": trend,
            "recent_alerts": alerts,
            "last_action_result": "",
            "diagnosis_made": self.diagnosis_made,
        }
        return observation

    # ---------------------------------------------------------------------
    # Public API used by env/environment.py
    # ---------------------------------------------------------------------
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply an action, update hidden state, compute reward and termination.
        Returns (observation, reward, done, info).
        """
        self.current_step += 1
        info = {}
        reward = 0.0
        done = False
        # -----------------------------------------------------------------
        # 1. Apply action effects on hidden state
        # -----------------------------------------------------------------
        action_type = action.get("type")
        target = action.get("target")
        # Helper to check if target refers to the root cause
        is_root = target == self.root_service
        # Record a textual description of what happened for the observation
        result_msg = ""
        # -----------------------------------------------------------------
        # Diagnosis action (no state change)
        # -----------------------------------------------------------------
        if action_type == "diagnose":
            if not self.diagnosis_made:
                self.diagnosis_made = True
                mode = action.get("failure_mode")
                if is_root and mode == self.root_failure:
                    reward += 8.0  # correct diagnosis
                else:
                    reward -= 2.0  # wrong diagnosis
                result_msg = f"Diagnosed {target} as {mode}."
            else:
                result_msg = "Diagnosis already made; no effect."
        # -----------------------------------------------------------------
        # Fix actions
        # -----------------------------------------------------------------
        elif action_type in {"restart_service", "rollback_deploy", "scale_up", "enable_circuit_breaker"}:
            if action_type == "restart_service":
                if is_root and self.root_failure in {"crashed", "memory_leak"}:
                    # full restore
                    self.true_health[target] = 1.0
                    # memory leak recurrence handling
                    if self.root_failure == "memory_leak":
                        self.memory_leak_counter = MEMORY_LEAK_RECURRENCE_STEPS
                else:
                    # no effect on health (penalty will be applied later)
                    pass
            elif action_type == "rollback_deploy":
                if is_root and self.root_failure == "bad_deploy":
                    self.true_health[target] = 1.0
                else:
                    pass
            elif action_type == "scale_up":
                if is_root and self.root_failure == "overloaded":
                    self.true_health[target] = 1.0
                else:
                    pass
            elif action_type == "enable_circuit_breaker":
                # isolation caps health at 0.75 for all downstream services
                for idx, svc in enumerate(SERVICES):
                    if PROPAGATION_MATRIX[SERVICES.index(target)][idx] > 0:
                        self.true_health[svc] = min(self.true_health[svc], 0.75)
                # BUG FIX: apply wrong‑service penalty if target not root
                if not is_root:
                    reward -= 2.0
                result_msg = f"Circuit breaker enabled on {target}."
            # generic fix penalty – handled later using lagged health
        # -----------------------------------------------------------------
        # No‑op action
        # -----------------------------------------------------------------
        elif action_type == "no_op":
            reward -= 0.5
            result_msg = "No‑op taken."
        # -----------------------------------------------------------------
        # Unknown action
        # -----------------------------------------------------------------
        else:
            result_msg = f"Unknown action type: {action_type}."
        # -----------------------------------------------------------------
        # 2. Propagation & memory leak recurrence handling
        # -----------------------------------------------------------------
        self._propagate()
        # handle memory leak recurrence if counter active
        if self.memory_leak_counter is not None:
            self.memory_leak_counter -= 1
            if self.memory_leak_counter == 0:
                # re‑apply memory leak degradation to root service
                self.true_health[self.root_service] = max(0.0, self.true_health[self.root_service] - 0.2)
                self.memory_leak_counter = None
        # -----------------------------------------------------------------
        # 3. Compute health delta for reward (using lagged health)
        # -----------------------------------------------------------------
        prev_lagged = self.health_history[-HEALTH_LAG_STEPS] if len(self.health_history) >= HEALTH_LAG_STEPS else {svc: 1.0 for svc in SERVICES}
        curr_lagged = {}
        for svc in SERVICES:
            # observation lag – health after this step will be visible next step
            curr_lagged[svc] = self.true_health[svc]
        health_delta = sum(curr_lagged[svc] - prev_lagged[svc] for svc in SERVICES) / len(SERVICES)
        # health delta reward (capped)
        if health_delta > 0:
            reward += min(health_delta * 2.0, 3.0)
        # -----------------------------------------------------------------
        # 4. Fix‑action penalties based on lagged health (BUG FIX)
        # -----------------------------------------------------------------
        if action_type in {"restart_service", "rollback_deploy", "scale_up", "enable_circuit_breaker"}:
            # use lagged health BEFORE the action (prev_lagged)
            lagged_health = prev_lagged.get(target, 1.0)
            if lagged_health > 0.85:
                reward -= 1.5  # penalty for fixing a healthy service
            # correct‑fix bonus (requires correct diagnosis flag)
            if is_root:
                reward += 10.0 if self.diagnosis_made else 6.0
            else:
                reward -= 2.0  # wrong‑service fix penalty
        # -----------------------------------------------------------------
        # 5. Termination check & success/ collapse rewards
        # -----------------------------------------------------------------
        system_health = sum(self.true_health.values()) / len(SERVICES)
        # success condition: health >=0.90 for 2 consecutive steps
        if system_health >= 0.90:
            # track consecutive success steps in info
            info.setdefault("success_steps", 0)
            info["success_steps"] += 1
        else:
            info["success_steps"] = 0
        # collapse condition
        if system_health <= 0.10:
            info.setdefault("collapse_steps", 0)
            info["collapse_steps"] += 1
        else:
            info["collapse_steps"] = 0
        # Determine done flag
        if info.get("success_steps", 0) >= 2:
            done = True
            reward += 20.0 + (self.max_steps - self.current_step) * 0.3
        elif info.get("collapse_steps", 0) >= 3:
            done = True
            reward -= 15.0
        elif self.current_step >= self.max_steps:
            done = True
        # -----------------------------------------------------------------
        # 6. Build observation for next step (includes last_action_result)
        # -----------------------------------------------------------------
        observation = self._generate_observation()
        observation["last_action_result"] = result_msg
        # store lagged health for next step
        self.health_history.append(copy.deepcopy(self.true_health))
        return observation, reward, done, info
