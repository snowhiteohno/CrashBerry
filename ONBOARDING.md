# CrashBerry — Complete Onboarding Guide

> **For anyone new to this project.** Read this top-to-bottom before touching any code.
> It explains *what* the project does, *why* each file exists, *how* every piece fits together,
> and *how* to run everything end-to-end on your own machine or Google Colab.

---

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [The Big Idea (Plain English)](#2-the-big-idea-plain-english)
3. [Repository Layout](#3-repository-layout)
4. [How the Environment Works](#4-how-the-environment-works)
5. [How the Agents Work](#5-how-the-agents-work)
6. [How the Demo Works](#6-how-the-demo-works)
7. [How Evaluation Works](#7-how-evaluation-works)
8. [How Training Works](#8-how-training-works)
9. [Running Locally](#9-running-locally)
10. [Running on Google Colab (Recommended)](#10-running-on-google-colab-recommended)
11. [Key Numbers and Thresholds](#11-key-numbers-and-thresholds)
12. [Common Errors and Fixes](#12-common-errors-and-fixes)
13. [Glossary](#13-glossary)

---

## 1. What is this project?

**CrashBerry** is an AI-powered incident response simulator.

It models a fake microservice-based system (like what you would see inside a company like Netflix
or Uber) that randomly breaks in different ways. An AI "agent" is then trained or tested to
diagnose the problem and fix it before the whole system collapses.

This project was built for a hackathon. The goal is to show that a large language model (LLM)
can be fine-tuned using **Reinforcement Learning (RL)** to act as a smart on-call engineer —
reading noisy system metrics, guessing the root cause of an outage, and applying the right fix.

---

## 2. The Big Idea (Plain English)

Imagine five microservices running a website:

```
api-gateway  →  auth-service  →  database  →  cache  →  worker
```

Every time an episode starts, one service **secretly breaks** in one of four ways:
- **crashed** — the service is completely down (health = 0)
- **memory_leak** — the service slowly degrades and even comes back after a restart
- **overloaded** — too many requests, needs more compute
- **bad_deploy** — a bad code deployment broke it, needs rollback

The agent can only see **noisy metrics** (CPU, error rate, latency, queue depth) — it cannot see
the true hidden state. It must figure out which service is broken and why, then apply the right fix.

The agent gets **rewards** for:
- Correctly diagnosing the root cause (+8 points)
- Applying the right fix to the right service (+10 points with prior diagnosis, +6 without)
- The system health improving after its action (up to +3 points per step)
- Resolving the incident quickly (+20 + time bonus)

The agent gets **penalties** for:
- Wrong diagnosis (-2), wrong service fix (-2), fixing a healthy service (-1.5)
- Doing nothing every step (-0.5 per no-op)
- Letting the system fully collapse (-15)

The whole episode ends after **20 steps** (or sooner if the system recovers or collapses).

---

## 3. Repository Layout

```
CrashBerry/
│
├── env/                        # The simulated environment
│   ├── __init__.py
│   ├── simulator.py            # Core simulation logic (hidden state, failure propagation)
│   ├── environment.py          # Gym-style wrapper around the simulator
│   └── grader.yaml             # Configuration constants (thresholds, service names, etc.)
│
├── agent/                      # Agent implementations
│   ├── __init__.py
│   └── model.py                # RandomAgent, HeuristicAgent, LLMAgent, GeminiAgent
│
├── tools/                      # Utility functions agents can call
│   ├── __init__.py
│   └── tools.py                # check_logs(), is_fix_action()
│
├── demo/                       # Gradio web interface
│   ├── __init__.py
│   └── app.py                  # Gradio UI code
│
├── eval/                       # Evaluation harness
│   ├── __init__.py
│   └── evaluate.py             # Runs N episodes and prints a summary table
│
├── run_demo.py                 # ← START HERE: root-level launcher for the demo
├── train.py                    # RL training loop (Qwen-1.5B + PPO via Unsloth + TRL)
├── metafinalfinalprd.md        # The original Product Requirements Document (PRD)
├── README.md                   # Short project overview
├── ONBOARDING.md               # ← This file
└── .gitignore
```

---

## 4. How the Environment Works

### Files: `env/simulator.py` and `env/environment.py`

There are two layers:

### Layer 1: `Simulator` (the engine)

`env/simulator.py` contains the raw simulation logic. It manages:

**Hidden State** (the agent cannot see these directly):
```python
self.true_health      # dict: {service: float 0.0–1.0}  — real health of each service
self.root_service     # which service secretly broke first
self.root_failure     # what type of failure it is
```

**On `reset()`:**
1. All services start at health = 1.0 (perfect).
2. One service is randomly chosen as the root cause.
3. A random failure mode is applied to that service.
4. The failure **propagates** to downstream services using the `PROPAGATION_MATRIX`.
5. The first observation is generated and returned.

**The Propagation Matrix:**
This is a 5x5 table that says "if service A is sick, how much does it hurt service B?"
For example, if `database` crashes (row index 2), it hurts `cache` by 30% and `worker` by 60%.
This is what makes the problem hard — you see multiple services degrading but only one is the
root cause.

**On `step(action)`:**
1. The action is applied (e.g., `restart_service` sets the target's health back to 1.0).
2. The propagation matrix is re-applied each step (a sick service keeps hurting its neighbors).
3. Memory leak recurrence: if you restart a memory-leaking service, after 4 steps it starts
   leaking again — the agent must act decisively.
4. Rewards are computed (see Section 2).
5. Termination is checked: success (health ≥ 0.90 for 2 consecutive steps) or collapse
   (health ≤ 0.10 for 3 steps) or max steps reached.
6. A new noisy observation is generated and returned.

**Observations are noisy and lagged:**
- Each metric (cpu, error_rate, latency_ms, queue_depth) has Gaussian noise (std = 0.15) added.
- Health values are lagged by 1 step — you see what happened last step, not right now.
- This simulates the real world where dashboards update slowly and sensors are imperfect.

**What the observation dict looks like:**
```python
{
    "step": 3,
    "max_steps": 20,
    "system_health_score": 0.62,
    "metrics": {
        "api-gateway": {"cpu": 0.85, "error_rate": 0.12, "latency_ms": 1200, "queue_depth": 45},
        "auth-service": {"cpu": 0.90, "error_rate": 0.08, "latency_ms": 800, "queue_depth": 30},
        "database":     {"cpu": 0.05, "error_rate": 0.93, "latency_ms": 1950, "queue_depth": 180},
        "cache":        {"cpu": 0.40, "error_rate": 0.55, "latency_ms": 1100, "queue_depth": 110},
        "worker":       {"cpu": 0.30, "error_rate": 0.65, "latency_ms": 1300, "queue_depth": 130},
    },
    "metric_trend": {"api-gateway": "stable", "database": "degrading", ...},
    "recent_alerts": ["database: health low (0.05)", "cache: health low (0.45)"],
    "last_action_result": "Diagnosed database as crashed.",
    "diagnosis_made": True,
}
```

### Layer 2: `IncidentResponseEnv` (the Gym wrapper)

`env/environment.py` wraps the `Simulator` in a clean OpenAI-Gym-style API:

```python
env = IncidentResponseEnv(seed=42)
obs = env.reset()                  # start a new episode
obs, reward, done, info = env.step(action)  # take one step
env.render()                       # print hidden state (for debugging)
```

The `seed` parameter makes episodes reproducible — same seed = same root cause, same random
noise sequence. Pass `seed=None` for random episodes.

---

## 5. How the Agents Work

### File: `agent/model.py`

All agents expose a single method: `agent.select_action(observation) → action_dict`.

An action dict looks like:
```python
{"type": "diagnose",         "target": "database", "failure_mode": "crashed"}
{"type": "restart_service",  "target": "database"}
{"type": "rollback_deploy",  "target": "auth-service"}
{"type": "scale_up",         "target": "worker"}
{"type": "enable_circuit_breaker", "target": "api-gateway"}
{"type": "no_op"}
```

### RandomAgent
Picks a completely random valid action every step. Used as the weakest possible baseline.
Expected performance: ~0% success rate.

### HeuristicAgent
A hand-crafted rule:
1. First step: pick the service with the highest CPU as the suspected culprit and diagnose it
   as "crashed" (the most common failure mode).
2. Every step after that: apply `restart_service` to that same service.

This is smarter than random but dumb — it always guesses "crashed" and never adapts.
Expected performance: ~20–30% success rate.

### LLMAgent
Calls the **Hugging Face Inference API** with a text prompt containing the observation JSON.
The model must output a valid JSON action. Requires:
- A Hugging Face account and API token (`HF_API_TOKEN` env variable).
- The chosen model to be accessible (gated models like Llama require license acceptance).

Default model: `meta-llama/Meta-Llama-3.1-8B-Instruct`  
Default temperature: `0.7` (randomness in generation — 0.0 = always the same output)

If the API call fails for any reason, it silently falls back to `no_op`.

### GeminiAgent ← Recommended for testing
Calls the **Google Gemini API** (`gemini-1.5-flash` by default). This is:
- **Free** (1 million tokens/month on the free tier)
- **No license agreement** required
- **Fast** and reliable
- Requires only a free API key from https://aistudio.google.com/app/apikey

Set the key in your environment:
```python
import os
os.environ["GOOGLE_API_KEY"] = "your_key_here"
```

The `GeminiAgent` uses a detailed system prompt that explains the valid action types and JSON
format, making it significantly better at producing valid actions than `LLMAgent`.

### Factory function
```python
from agent.model import get_agent

agent = get_agent("random")    # RandomAgent
agent = get_agent("heuristic") # HeuristicAgent
agent = get_agent("llm")       # LLMAgent
agent = get_agent("gemini")    # GeminiAgent
```

---

## 6. How the Demo Works

### Files: `demo/app.py` and `run_demo.py`

The demo is a **Gradio web app** — it gives you a browser UI where you can:
1. Pick an agent from a dropdown (random / heuristic / llm / gemini).
2. Click "Run Episode" to run one full episode (max 15 steps).
3. See every step's action, reward, and observation printed out.
4. See the total reward and how the episode ended (success / collapse / max steps).

**Concurrency safety:** A threading `Semaphore` limits the demo to one episode at a time.
If you click "Run Episode" while another is already running, you get a "busy" message.

**How to launch:**

*Local machine:*
```bash
cd CrashBerry
python run_demo.py
# Open http://127.0.0.1:7860 in your browser
```

*Google Colab (gives a public share link):*
```python
%cd /content/CrashBerry
!python run_demo.py
# Click the gradio.live link that appears
```

**Why `run_demo.py` and not `python demo/app.py` directly?**

`run_demo.py` lives at the **project root**. When Python runs it, the project root is
automatically on the import path, so `from env.environment import ...` works immediately.

Running `demo/app.py` directly requires Python to also know where the project root is, which
is more fragile. Always use `run_demo.py`.

---

## 7. How Evaluation Works

### File: `eval/evaluate.py`

The evaluation harness runs an agent through N episodes and prints summary statistics.

```bash
# From the project root:
python eval/evaluate.py random    -n 100   # 100 episodes with the RandomAgent
python eval/evaluate.py heuristic -n 100
python eval/evaluate.py llm       -n 20
```

**Output looks like:**
```
--- Evaluation Summary ---
Episodes: 100
Success Rate: 15.00%
Diagnosis Accuracy: 22.00%
Mean Steps to Resolution: 19.45
Average Reward per Episode: 42.30
```

**What each metric means:**
- **Success Rate:** % of episodes where system health stayed ≥ 0.90 for 2+ consecutive steps.
- **Diagnosis Accuracy:** % of episodes where the agent's `diagnose` action named the correct
  service AND the correct failure mode.
- **Mean Steps to Resolution:** Average number of steps before the episode ended.
  Lower = faster resolution. If always 20, the agent ran out of steps every time.
- **Average Reward per Episode:** Total cumulative reward averaged across episodes.

---

## 8. How Training Works

### File: `train.py`

The training script fine-tunes **Qwen-1.5B** (a small, open-source LLM) using **PPO**
(Proximal Policy Optimization — a Reinforcement Learning algorithm) from the **TRL** library.
**Unsloth** is used for 4-bit quantization to make it fit on a free Colab T4 GPU.

**How the RL loop works:**
1. Reset the environment to get an initial observation.
2. Encode the observation as text and feed it to the model.
3. The model generates a JSON action string.
4. Parse the JSON and send it to `env.step()`.
5. The reward from the environment is used as the RL reward signal.
6. PPO updates the model weights to make high-reward actions more likely in the future.
7. Repeat until the episode ends, then start a new episode.

**Run training in Colab (T4 GPU required):**
```python
%cd /content/CrashBerry
!python train.py --epochs 10 --lr 1e-5
```

**Checkpoints** are saved after every epoch to `./trained_models/checkpoint_epoch_N/`.  
The final model is saved to `./trained_models/final_model/`.

**Key training arguments:**
```
--model-name    HuggingFace model ID (default: Qwen/Qwen1.5-1.8B)
--epochs        Number of training episodes (default: 3)
--lr            Learning rate (default: 1e-5)
--seed          Random seed for reproducibility (default: 42)
--output-dir    Where to save model checkpoints (default: ./trained_models)
```

---

## 9. Running Locally

### Prerequisites
```bash
pip install gradio requests pyyaml huggingface_hub google-generativeai
```

For training only (requires a GPU):
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl accelerate torch
```

### Quick start
```bash
# Clone the repo
git clone https://github.com/snowhiteohno/CrashBerry.git
cd CrashBerry

# Run 10 heuristic evaluation episodes (no API key needed)
python eval/evaluate.py heuristic -n 10

# Start the Gradio demo
python run_demo.py
```

### To use the Gemini agent locally
```bash
# Windows PowerShell
$env:GOOGLE_API_KEY = "your_key_here"
python run_demo.py

# Mac / Linux
GOOGLE_API_KEY="your_key_here" python run_demo.py
```

---

## 10. Running on Google Colab (Recommended)

Colab has free internet access and free GPUs — ideal for LLM agent testing and training.

### Step-by-step

**Cell 1 — Clone and install:**
```python
!git clone https://github.com/snowhiteohno/CrashBerry.git
%cd /content/CrashBerry
!pip install -q gradio requests pyyaml huggingface_hub google-generativeai
```

**Cell 2 — Set your Gemini API key** (get it free from https://aistudio.google.com/app/apikey):
```python
import os
os.environ["GOOGLE_API_KEY"] = "your_key_here"
```

**Cell 3 — Launch the demo:**
```python
%cd /content/CrashBerry
!python run_demo.py
```
Click the `gradio.live` public URL that appears. Select **gemini** in the dropdown. ✅

**Cell 4 — Run evaluation (optional):**
```python
!python eval/evaluate.py heuristic -n 50
```

**Cell 5 — Run training (needs T4 GPU runtime):**
```python
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" trl accelerate
!python train.py --epochs 5
```

---

## 11. Key Numbers and Thresholds

| Parameter | Value | Where set |
|-----------|-------|-----------|
| Number of services | 5 | `env/simulator.py` → `SERVICES` |
| Failure modes | 4 | `env/simulator.py` → `FAILURE_MODES` |
| Max steps per episode | 20 | `env/grader.yaml` → `max_steps` |
| Demo step cap | 15 | `demo/app.py` → `_DEMO_MAX_STEPS` |
| Observation noise (Gaussian std) | 0.15 | `env/simulator.py` → `NOISE_STD` |
| Health observation lag | 1 step | `env/simulator.py` → `HEALTH_LAG_STEPS` |
| Memory leak recurrence delay | 4 steps | `env/simulator.py` → `MEMORY_LEAK_RECURRENCE_STEPS` |
| Success threshold | 0.90 for 2 steps | `env/grader.yaml` → `success_threshold` |
| Collapse threshold | 0.10 for 3 steps | `env/grader.yaml` → `failure_threshold` |
| Correct diagnosis reward | +8.0 | `env/simulator.py` line 184 |
| Wrong diagnosis penalty | -2.0 | `env/simulator.py` line 186 |
| Correct fix + prior diagnosis | +10.0 | `env/simulator.py` line 268 |
| Correct fix without diagnosis | +6.0 | `env/simulator.py` line 268 |
| Wrong service fix penalty | -2.0 | `env/simulator.py` line 270 |
| No-op penalty | -0.5/step | `env/simulator.py` line 228 |
| Collapse penalty | -15.0 | `env/simulator.py` line 294 |
| Success bonus | +20.0 + time | `env/simulator.py` line 291 |
| LLM default temperature | 0.7 | `agent/model.py` → `LLMAgent.__init__` |
| Gemini default model | gemini-1.5-flash | `agent/model.py` → `GeminiAgent.__init__` |
| Training model | Qwen/Qwen1.5-1.8B | `train.py` argument default |

---

## 12. Common Errors and Fixes

### `ModuleNotFoundError: No module named 'env'`
**Cause:** You ran `python demo/app.py` directly instead of using the root launcher.  
**Fix:** Always run `python run_demo.py` from the project root directory.

### `Error creating agent: GOOGLE_API_KEY environment variable not set`
**Cause:** You selected the **gemini** agent but haven't set the API key.  
**Fix:** Get a free key at https://aistudio.google.com/app/apikey and set:
```python
import os; os.environ["GOOGLE_API_KEY"] = "your_key_here"
```

### `Error creating agent: HF_MODEL_ID environment variable must be set for LLMAgent`
**Cause:** Old cached version of `model.py` — the server was not restarted after the fix.  
**Fix:** Restart the Gradio server. Also consider using **gemini** instead of **llm**.

### LLM agent always outputs the same thing / runs instantly
**Cause:** The HuggingFace API call is silently failing (DNS error, auth error, gated model).  
The agent falls back to `no_op` every step → 15 identical steps in under a second.  
**Fix:** Check terminal output for `[LLMAgent Debug]` lines. Switch to **gemini** agent.

### `Failed to resolve 'api.huggingface.co' (Errno 11001)`
**Cause:** Your network blocks outbound connections to huggingface.co (corporate firewall, VPN).  
**Fix:** Use Google Colab (unrestricted internet) or switch to **gemini** agent.

### Demo shows `http://127.0.0.1:7860` in Colab but can't open it
**Cause:** Colab runs in the cloud — localhost URLs only work on that remote machine.  
**Fix:** Run via `run_demo.py` which uses `share=True` automatically, generating a public `gradio.live` URL.

---

## 13. Glossary

| Term | Meaning |
|------|---------|
| **Episode** | One full run from `env.reset()` until `done=True`. Like one shift of an on-call engineer. |
| **Step** | One action taken by the agent inside an episode. |
| **Observation** | The dict of noisy metrics the agent sees — its only window into the system state. |
| **Hidden state** | The true health values and root cause that the agent cannot see directly. |
| **Root cause** | The one service that originally broke, triggering cascading failures in others. |
| **Propagation** | When one sick service makes others sick via the dependency matrix. |
| **PPO** | Proximal Policy Optimization — the RL algorithm used to train the LLM agent. |
| **Unsloth** | A library that makes training LLMs faster by applying 4-bit quantization and flash attention. |
| **TRL** | Transformer Reinforcement Learning — a HuggingFace library providing PPOTrainer. |
| **Gradio** | A Python library that builds browser-based UIs for ML models with a few lines of code. |
| **Temperature** | Controls how random an LLM's output is. 0.0 = always the same output, 1.0 = very random. |
| **Semaphore** | A concurrency lock that ensures only one episode runs at a time in the demo. |
| **Health lag** | Observations are 1 step behind reality — you see last step's health, not the current one. |
| **Grader YAML** | `env/grader.yaml` — a configuration file that defines thresholds and service names. |
