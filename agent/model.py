"""Agent implementations for the Incident Response environment.
Three agents are provided:
1. RandomAgent – picks a random valid action each step.
2. HeuristicAgent – uses the highest CPU metric as a guess for the root cause,
   then performs a diagnose followed by the appropriate fix.
3. LLMAgent – calls a HuggingFace Inference API model (model id supplied via
   the HF_API_TOKEN environment variable). The model receives a JSON string of
   the observation and must output a JSON action matching the PRD schema.
All agents expose a ``select_action(observation)`` method that returns a dict
compatible with the environment's ``step`` method.
"""
import os
import json
import random
from typing import Dict, Any, List
import requests

try:
    from huggingface_hub import get_token
except ImportError:
    get_token = lambda: None

from env.environment import IncidentResponseEnv
from tools.tools import is_fix_action, check_logs

# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------

def _available_actions(observation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate a list of possible actions given the current observation.
    This mirrors the action space defined in the PRD.
    """
    services = list(observation["metrics"].keys())
    actions = []
    # diagnose (only once per episode – handled by agent)
    actions.append({"type": "diagnose", "target": None, "failure_mode": None})
    for svc in services:
        actions.append({"type": "restart_service", "target": svc})
        actions.append({"type": "rollback_deploy", "target": svc})
        actions.append({"type": "scale_up", "target": svc})
        actions.append({"type": "enable_circuit_breaker", "target": svc})
        actions.append({"type": "check_logs", "target": svc})
    actions.append({"type": "no_op"})
    return actions

# ---------------------------------------------------------------------
# Random Agent
# ---------------------------------------------------------------------
class RandomAgent:
    """Selects a completely random valid action each step.
    The ``diagnose`` action is also random – it may be called multiple times
    but the environment only scores the first one.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        actions = _available_actions(observation)
        action = self.rng.choice(actions)
        # fill required fields for diagnose if chosen
        if action["type"] == "diagnose":
            services = list(observation["metrics"].keys())
            action["target"] = self.rng.choice(services)
            action["failure_mode"] = self.rng.choice(["crashed", "memory_leak", "overloaded", "bad_deploy"])
        return action

# ---------------------------------------------------------------------
# Heuristic Agent
# ---------------------------------------------------------------------
class HeuristicAgent:
    """A simple heuristic:
    - Choose the service with the highest CPU metric as the suspected root cause.
    - Diagnose it with the most common failure mode (crashed).
    - Apply the appropriate fix based on the guessed failure mode.
    - If diagnosis already made, directly attempt the correct fix.
    """

    def __init__(self):
        self.diagnosed = False
        self.guessed_service = None
        self.guessed_mode = None

    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        # Determine service with highest CPU metric (noisy but a reasonable proxy)
        metrics = observation["metrics"]
        highest_cpu_service = max(metrics.items(), key=lambda x: x[1]["cpu"])[0]
        if not self.diagnosed:
            # Diagnose with a default mode – we pick "crashed" as heuristic.
            self.guessed_service = highest_cpu_service
            self.guessed_mode = "crashed"
            self.diagnosed = True
            return {"type": "diagnose", "target": self.guessed_service, "failure_mode": self.guessed_mode}
        else:
            # Choose fix based on guessed mode
            if self.guessed_mode == "crashed" or self.guessed_mode == "memory_leak":
                return {"type": "restart_service", "target": self.guessed_service}
            if self.guessed_mode == "overloaded":
                return {"type": "scale_up", "target": self.guessed_service}
            if self.guessed_mode == "bad_deploy":
                return {"type": "rollback_deploy", "target": self.guessed_service}
            # fallback
            return {"type": "no_op"}

# ---------------------------------------------------------------------
# LLM Agent (HF Inference API)
# ---------------------------------------------------------------------
class LLMAgent:
    """Calls a HuggingFace Inference API model to decide actions.
    The model ID and optional token are read from environment variables:
    - ``HF_MODEL_ID`` – required (e.g., "meta-llama/Llama-3.1-8B-Instruct")
    - ``HF_API_TOKEN`` – optional if the model is public.
    The model receives the observation JSON as a prompt and must output a
    JSON action matching the PRD specification.
    """

    def __init__(self, model_id: str | None = None, token: str | None = None, temperature: float = 0.7):
        self.model_id = model_id or os.getenv("HF_MODEL_ID") or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.token = token or os.getenv("HF_API_TOKEN") or get_token()
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.temperature = temperature
        self.session = requests.Session()

    def _call_model(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        # Build a simple text prompt – the model is expected to output JSON.
        prompt = (
            "You are controlling an incident response agent. Given the following observation, "
            "output a JSON action. Use the exact schema from the PRD.\n\nObservation:\n"
            f"{json.dumps(observation, indent=2)}\n\nAction:"
        )
        payload = {
            "inputs": prompt,
            "parameters": {"temperature": self.temperature, "max_new_tokens": 200},
            "options": {"wait_for_model": True},
        }
        url = f"https://api.huggingface.co/models/{self.model_id}"
        response = self.session.post(url, headers=self.headers, json=payload, timeout=30)
        response.raise_for_status()
        # HF returns a list of generated strings; we assume the first.
        generated = response.json()
        if isinstance(generated, list):
            generated = generated[0]
        if isinstance(generated, dict) and "generated_text" in generated:
            generated_str = generated["generated_text"]
        elif isinstance(generated, dict) and "error" in generated:
            raise ValueError(f"API Error: {generated['error']}")
        else:
            generated_str = str(generated)
            
        # The model may return a JSON string possibly with surrounding text.
        try:
            # Find first '{' and last '}' to extract JSON.
            start = generated_str.find('{')
            end = generated_str.rfind('}')
            if start == -1 or end == -1:
                raise ValueError("No JSON object found in output")
            json_str = generated_str[start : end + 1]
            action = json.loads(json_str)
        except Exception as e:
            print(f"[LLMAgent Debug] Parse failed: {e}\nRaw output: {generated_str}")
            raise ValueError(f"Failed to parse model output as JSON: {e}\nRaw output: {generated_str}")
        return action

    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._call_model(observation)
        except Exception as e:
            print(f"[LLMAgent Debug] Action fallback to no_op due to error: {e}")
            # On any failure fall back to a safe no_op to keep the episode alive.
            return {"type": "no_op"}

# ---------------------------------------------------------------------
# Export a simple factory for external callers
# ---------------------------------------------------------------------
def get_agent(name: str, **kwargs):
    """Factory returning an agent instance.
    ``name`` can be "random", "heuristic", or "llm".
    ``kwargs`` are passed to the constructor of the chosen class.
    """
    name = name.lower()
    if name == "random":
        return RandomAgent(**kwargs)
    if name == "heuristic":
        return HeuristicAgent(**kwargs)
    if name == "llm":
        return LLMAgent(**kwargs)
    raise ValueError(f"Unknown agent name: {name}")
