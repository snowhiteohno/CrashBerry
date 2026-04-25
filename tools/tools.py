"""Utility tools for the Incident Response environment.
Implements the `check_logs` function and helpers used by agents.
The implementations follow the PRD action‑effect table and provide
simple deterministic hints for each service based on the hidden state.
"""
import random
from typing import Dict, List

# Simple deterministic log hints seeded by service and failure mode.
# In a real system this would pull from hidden state; here we provide a
# lightweight placeholder that agents can call.
_LOG_HINTS = {
    "api-gateway": ["high latency observed", "increased error rate"],
    "auth-service": ["authentication failures rising", "CPU spikes"],
    "database": ["query timeouts", "connection errors"],
    "cache": ["cache miss rate increased", "slow responses"],
    "worker": ["job queue backlog", "worker crashes"],
}

def check_logs(service: str, root_service: str = None, failure_mode: str = None) -> List[str]:
    """Return noisy, partial log hints for `service`.
    If `service` is the hidden root cause, we include 2‑3 strong hints plus
    random noise; otherwise only noise.
    The function does not access hidden state directly – agents are expected
    to call this with the environment's hidden parameters (which they do not
    know). For the purpose of the demo we simulate the effect using the
    provided `root_service` and `failure_mode` arguments.
    """
    hints = []
    # always add a small amount of noise
    noise = ["log entry at {}".format(random.randint(1000, 9999)) for _ in range(2)]
    hints.extend(noise)
    if service == root_service:
        # add stronger hints based on failure mode
        base = _LOG_HINTS.get(service, [])
        strong = random.sample(base, min(2, len(base)))
        hints.extend(strong)
        # add a hint specific to failure mode
        hints.append(f"detected {failure_mode} symptom")
    return hints

def is_fix_action(action: Dict) -> bool:
    """Return True if the action type is a fix action.
    Fix actions are: restart_service, rollback_deploy, scale_up, enable_circuit_breaker.
    """
    return action.get("type") in {"restart_service", "rollback_deploy", "scale_up", "enable_circuit_breaker"}
