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
# Gemini Agent (Google AI Studio – free tier)
# ---------------------------------------------------------------------
class GeminiAgent:
    """Uses the Google Gemini API to decide actions.
    Requires a free API key from https://aistudio.google.com/app/apikey
    Set it as the environment variable GOOGLE_API_KEY.

    In Colab:
        import os; os.environ['GOOGLE_API_KEY'] = 'your_key_here'
    or use Colab Secrets (lock icon in the left sidebar).
    """

    def __init__(self, model_id: str = "gemini-2.0-flash", temperature: float = 0.7):
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Run: pip install -q google-genai"
            )
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        self._client = genai.Client(api_key=api_key)
        self._model_id = model_id
        self._config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=256,
        )
        self._system_prompt = (
            "You are an incident response agent controlling a microservice system. "
            "Given an observation JSON, output ONLY a valid JSON action with no extra text.\n"
            "Valid action types: diagnose, restart_service, rollback_deploy, scale_up, "
            "enable_circuit_breaker, check_logs, no_op.\n"
            "For 'diagnose', include 'target' (service name) and 'failure_mode' "
            "(one of: crashed, memory_leak, overloaded, bad_deploy).\n"
            "For all other types (except no_op), include 'target' (service name).\n"
            "Example: {\"type\": \"diagnose\", \"target\": \"auth\", \"failure_mode\": \"crashed\"}"
        )

    def _call_model(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"{self._system_prompt}\n\nObservation:\n"
            f"{json.dumps(observation, indent=2)}\n\nAction JSON:"
        )
        response = self._client.models.generate_content(
            model=self._model_id,
            contents=prompt,
            config=self._config,
        )
        text = response.text.strip()
        # Extract JSON from the response
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            raise ValueError(f"No JSON found in Gemini response: {text}")
        action = json.loads(text[start:end + 1])
        return action

    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._call_model(observation)
        except Exception as e:
            print(f"[GeminiAgent Debug] Fallback to no_op: {e}")
            raise  # re-raise so the demo UI catches and displays the real error

# ---------------------------------------------------------------------
# OpenAI Compatible Agent (OpenAI, Groq, etc.)
# ---------------------------------------------------------------------
class OpenAIAgent:
    """Uses any OpenAI-compatible API (OpenAI, Groq, etc.)
    Set OPENAI_API_KEY and optionally OPENAI_BASE_URL.
    """

    def __init__(self, model_id: str = "gpt-4o-mini", temperature: float = 0.7):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") # Can be used for Groq, etc.
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
            
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model_id = model_id
        self._temperature = temperature
        self._system_prompt = (
            "You are an incident response agent. Output ONLY a valid JSON action.\n"
            "Valid action types: diagnose, restart_service, rollback_deploy, scale_up, "
            "enable_circuit_breaker, check_logs, no_op.\n"
            "Example: {\"type\": \"diagnose\", \"target\": \"auth\", \"failure_mode\": \"crashed\"}"
        )

    def _call_model(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": f"Observation: {json.dumps(observation)}"}
            ],
            temperature=self._temperature,
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)

    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._call_model(observation)
        except Exception as e:
            print(f"[OpenAIAgent Debug] Error: {e}")
            raise

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
    if name == "gemini":
        return GeminiAgent(**kwargs)
    if name == "openai":
        return OpenAIAgent(**kwargs)
    raise ValueError(f"Unknown agent name: {name}")
