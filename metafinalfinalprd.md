# Product Requirements Document
## Stateful Incident Response Environment for LLM Agent Training

---

## 1. The Problem

Most agent benchmarks are broken in the same way: they reward outcomes, not reasoning.

An agent that guesses the right answer by trying all five options scores the same as one that diagnosed the root cause correctly on the first attempt. There is no way to tell them apart from the reward signal alone. This means training on outcome-only rewards produces agents that learn shortcuts — not causal reasoning, not persistent world models, not belief updating. Just pattern matching dressed up as intelligence.

This is the documented gap in LLM agent evaluation that this project directly addresses.

---

## 2. What We Built

A **partially observable professional task environment** where an LLM agent must act as an on-call engineer responding to a live production incident.

Five microservices are running. One has silently failed. Its failure propagates symptoms through a fixed dependency graph to downstream services, creating noisy, indirect evidence. The agent cannot see what broke. It must:

1. Read degraded metrics and alerts
2. Actively gather information using tools
3. **Explicitly commit to a diagnosis** — declaring the root cause service and failure mode before acting
4. Apply the correct fix
5. Confirm recovery

The critical design decision: **diagnosis is scored separately from the fix.** A lucky agent that stumbles onto the correct service without reasoning scores lower than one that diagnosed correctly and then acted. This makes reasoning quality measurable, trainable, and impossible to shortcut.

This directly targets what the problem statement calls *"real hard work instead of exploiting shortcuts to arrive at the desired outcome."*

---

## 3. Why This Environment

### The research backing

Current RL training for LLMs works well in domains with verifiable outcome rewards — math, code. It breaks down in professional task domains because:

- Outcome rewards cannot distinguish good reasoning from lucky guessing
- Agents learn to exploit reward signal transitions rather than solve the underlying task
- There is no intermediate supervision signal for reasoning quality

### What we add

A `diagnose(service, failure_mode)` action that creates an **explicit, scored reasoning checkpoint** mid-episode. This is a lightweight process reward model baked directly into the environment — not a separate reward model that can be gamed, but a structural constraint on the agent loop itself.

### What this enables

- Measurable separation between reasoning agents and brute-force agents
- Dense reward signal that reflects actual problem-solving steps
- An environment where causal reasoning and belief updating are *required*, not incidentally useful

---

## 4. Environment Design

### 4.1 The World

A production system of 5 microservices with a fixed dependency structure:

| Service | Role |
|---|---|
| `api-gateway` | Entry point, routes all traffic |
| `auth-service` | Handles authentication |
| `database` | Primary data store |
| `cache` | Redis-style caching layer |
| `worker` | Async job processing |

At the start of each episode, one service fails silently. The failure mode is sampled from four types. Symptoms propagate through the dependency graph. The agent sees only the downstream effects — never the root cause directly.

### 4.2 Hidden State

The agent cannot observe:

- Which service is the root cause
- What the failure mode is
- True per-service health values (only noisy approximations)

### 4.3 Visible Observation

At each step, the agent receives:

```json
{
  "step": 4,
  "max_steps": 20,
  "system_health_score": 0.43,
  "metrics": {
    "api-gateway":  { "cpu": 0.61, "error_rate": 0.18, "latency_ms": 420,  "queue_depth": 80  },
    "auth-service": { "cpu": 0.45, "error_rate": 0.09, "latency_ms": 210,  "queue_depth": 12  },
    "database":     { "cpu": 0.91, "error_rate": 0.51, "latency_ms": 1800, "queue_depth": 340 },
    "cache":        { "cpu": 0.38, "error_rate": 0.04, "latency_ms": 18,   "queue_depth": 3   },
    "worker":       { "cpu": 0.72, "error_rate": 0.22, "latency_ms": 640,  "queue_depth": 160 }
  },
  "metric_trend": {
    "api-gateway":  "degrading",
    "auth-service": "stable",
    "database":     "degrading",
    "cache":        "stable",
    "worker":       "degrading"
  },
  "recent_alerts": [
    "database: query timeout spike detected",
    "worker: job queue backlog growing"
  ],
  "last_action_result": "restarted auth-service: no significant change observed",
  "diagnosis_made": false
}
```

**Noise model:** ±15% Gaussian noise on all metrics. Health updates lag 1 step behind true state. The agent must reason under uncertainty — not read off ground truth.

**`metric_trend`** — computed from delta between current and previous true health. Values: `"degrading"`, `"stable"`, `"recovering"`. Gives temporal signal without exposing hidden state.

**`last_action_result`** — natural language description of what happened after the last action. Noisy but informative. Forces the agent to update its beliefs based on feedback.

### 4.4 Failure Modes and Required Fixes

| Failure Mode | Correct Fix | Notes |
|---|---|---|
| `crashed` | `restart_service` | Single action, full restore |
| `memory_leak` | `restart_service` | Temporary fix; recurs after 4 steps — agent must detect and stabilize |
| `overloaded` | `scale_up` | Restart has no effect; adds noise |
| `bad_deploy` | `rollback_deploy` | Restart actively worsens health (-0.1) |

### 4.5 Allowed Actions

| Action | Type | Effect |
|---|---|---|
| `check_logs(service)` | Information | Returns noisy, partial log hints seeded by `(service, failure_mode, step)`. Root cause service: 2–3 weak hints + noise. Healthy service: noise only. No single log is conclusive. |
| `diagnose(service, failure_mode)` | Reasoning checkpoint | No state change. Scores reasoning quality. One-shot — only first call scored. Tracked via `diagnosis_made` flag. |
| `restart_service(service)` | Fix | Full restore only if failure mode is `crashed` or `memory_leak` and target is root cause |
| `rollback_deploy(service)` | Fix | Full restore only if failure mode is `bad_deploy` and target is root cause |
| `scale_up(service)` | Fix | Full restore only if failure mode is `overloaded` and target is root cause |
| `enable_circuit_breaker(service)` | Isolation | Stops propagation from target; health capped at 0.75; cannot fully resolve on its own |
| `no_op()` | Passive | No state change; penalized |

**Action format:**
```json
{ "type": "diagnose", "target": "database", "failure_mode": "memory_leak" }
{ "type": "restart_service", "target": "database" }
{ "type": "check_logs", "target": "worker" }
```

### 4.6 Propagation Matrix

Fixed across all episodes. Only root cause and failure mode are sampled at reset. Fully deterministic given a seed.

```
              api-gateway  auth  database  cache  worker
api-gateway  [   1.0,      0.3,    0.0,    0.0,   0.0  ]
auth-service [   0.4,      1.0,    0.0,    0.0,   0.0  ]
database     [   0.5,      0.2,    1.0,    0.3,   0.6  ]
cache        [   0.2,      0.0,    0.1,    1.0,   0.3  ]
worker       [   0.3,      0.0,    0.0,    0.1,   1.0  ]
```

`matrix[i][j]` = fraction of service `i`'s degradation that bleeds to service `j` per step.

### 4.7 Termination

| Condition | Result |
|---|---|
| `system_health_score >= 0.90` for 2 consecutive steps | Success |
| `system_health_score <= 0.10` for 3 consecutive steps | Collapse |
| `step >= 20` | Timeout |

---

## 5. Reward Function

Reward priority (enforced structurally): **success > diagnosis > fix > health delta**

This ordering ensures an agent cannot compensate for bad reasoning with lucky health recovery.

| Priority | Signal | Value | Condition |
|---|---|---|---|
| 1 | Success | `+20.0` | Health ≥ 0.90 for 2 consecutive steps |
| 1 | Efficiency bonus | `+(max_steps - step) * 0.3` | Awarded on success; faster = higher |
| 2 | Diagnosis correct | `+8.0` | `diagnose()` matches hidden state exactly |
| 2 | Diagnosis wrong | `-2.0` | `diagnose()` called with wrong service or mode |
| 3 | Fix correct + diagnosed | `+10.0` | Correct fix on root cause after correct diagnosis |
| 3 | Fix correct, not diagnosed | `+6.0` | Correct fix on root cause without prior diagnosis |
| 3 | Fix wrong | `-2.0` | Fix action on non-root-cause service |
| 4 | Health delta | `+Δhealth * 2.0` | Capped at `+3.0` per step |
| — | Invalid action | `-1.5` | Fix on a service with lagged health > 0.85 |
| — | Wasted step | `-0.5` | `no_op()` called |
| — | Cascade collapse | `-15.0` | Terminal collapse |

**The +10 vs +6 gap is the core signal.** A brute-force agent that tries all five services accrues `-2.0 × 4` wrong-service penalties before hitting the right one. It cannot profit from spam. A reasoning agent that diagnoses first and fixes correctly gets `+8.0 + 10.0 = 18.0` before success reward. The gap in episode reward between these two strategies is measurable and grows with training.

```python
reward = 0.0

# Priority 2: Diagnosis (one-shot)
if action.type == "diagnose" and not diagnosis_made:
    if action.target == root_cause_service and action.failure_mode == root_failure_mode:
        reward += 8.0
        diagnosis_was_correct = True
    else:
        reward -= 2.0
    diagnosis_made = True

# Priority 3: Fix
if is_fix_action(action):
    if action.target == root_cause_service:
        reward += 10.0 if diagnosis_was_correct else 6.0
    else:
        reward -= 2.0

# Penalty: fixing healthy service (uses lagged health, what agent could observe)
if is_fix_action(action) and last_true_health[action.target] > 0.85:
    reward -= 1.5

# Priority 4: Health delta (capped)
if health_delta > 0:
    reward += min(health_delta * 2.0, 3.0)

# Penalties
if action.type == "no_op":
    reward -= 0.5

# Priority 1: Terminal
if done:
    if success:
        reward += 20.0 + (max_steps - step) * 0.3
    elif collapsed:
        reward -= 15.0
```

---

## 6. Agent Loop

Expected behavior per episode:

```
observe()       → read metrics, trends, alerts
check_logs()    → gather noisy hints across 1–3 services
diagnose()      → explicitly commit to root cause + failure mode
fix()           → apply correct action to correct service
observe()       → confirm health recovering
               → if memory_leak: detect recurrence, stabilize
done            → health ≥ 0.90 for 2 steps
```

---

## 7. OpenEnv Implementation

### 7.1 Environment Interface

```python
class IncidentResponseEnv:
    def reset(self, seed=None) -> Observation
    def step(self, action: dict) -> tuple[Observation, float, bool, dict]
    def render(self) -> str
    def seed(self, seed: int)
```

### 7.2 Grader Config

```yaml
env_id: incident-response-v1
score_range: [0, 1]
grader_count: 3
max_steps: 20
success_threshold: 0.90
failure_threshold: 0.10
services: [api-gateway, auth-service, database, cache, worker]
failure_modes: [memory_leak, crashed, overloaded, bad_deploy]
noise_std: 0.15
health_lag_steps: 1
```

---

## 8. Hugging Face Integration

HF is not just hosting. It is the demo layer, the evidence layer, and the distribution layer.

| Component | What it does |
|---|---|
| **Gradio Space** | Live interactive demo. Judge clicks "Run Episode," watches agent read logs, diagnose, fix, sees reward accumulate in real time. Random vs trained agent toggle. |
| **HF Dataset** | Published episode rollouts — full `(observation, action, reward)` traces. Inspectable without running anything. Signals research credibility. |
| **Trained model on HF Hub** | Qwen 1.5B fine-tuned on the environment. Model card includes before/after behavioral traces, reward curve, loss curve. |
| **Leaderboard on Space** | Three rows: random agent, heuristic agent, trained agent. Makes improvement instantly legible to judges. |

### 8.1 HF Inference API Token Budget

The hackathon provides $30 in HF Inference API credits. This is sufficient but not generous — allocation must be deliberate.

**Token cost model:**

| Use Case | Est. Cost per Episode | Notes |
|---|---|---|
| Live demo agent (judge-facing) | ~$0.001 | ~500 tokens × 5–10 LLM calls |
| Training loop | $0.00 | Never use HF tokens for training |
| `check_logs()` generation | $0.00 | Keep templated — spend tokens where judges see them |

At $0.001 per episode, $30 supports ~30,000 episodes theoretically. Real constraint is concurrent judge traffic during the finale — unpredictable and potentially fast-draining.

**Allocation rules:**

1. **Training stays on free Colab** — Qwen 1.5B runs locally via Unsloth. Zero token spend.
2. **`check_logs()` stays templated** — seeded deterministic strings, no API call.
3. **HF tokens are exclusively for the Gradio demo agent** — the live LLM the judge interacts with.
4. **Demo model is Llama 3.1 8B or Mistral 7B via HF Inference API** — stronger than the trained model, makes the live demo compelling.

**Safeguards (mandatory):**

```python
# In app.py — prevent budget drain during concurrent judge traffic
import threading
semaphore = threading.Semaphore(1)  # max 1 concurrent episode

def run_episode():
    if not semaphore.acquire(blocking=False):
        return "Demo is busy — another episode is running. Try again in 30 seconds."
    try:
        # run episode
    finally:
        semaphore.release()
```

- Max 1 concurrent episode via semaphore
- Max 15 steps per demo episode (vs 20 in training) to reduce token spend
- Add a visible "tokens remaining" estimate on the Space if possible
- If budget runs low before finale ends: fall back to heuristic agent for demo, keep trained model results as static evidence

---

## 9. Training Setup

| Component | Choice | Reason |
|---|---|---|
| Base model | Qwen 1.5B or Gemma 2B | Fits free Colab T4; Unsloth optimized |
| Training framework | HuggingFace TRL (GRPO) | Matches hackathon stack requirement |
| Efficient inference | Unsloth | Required for free-tier compute |
| Training environment | Google Colab free tier | T4 GPU, sufficient for 1.5–2B models |
| Episodes per run | 500–1000 | Enough to show reward curve movement |
| Token spend during training | $0.00 | Training is fully local — no HF API calls |

**Two-model strategy:**

- **Training model** — Qwen 1.5B, trained locally on Colab, weights pushed to HF Hub. This is the model whose improvement you measure and report.
- **Demo model** — Llama 3.1 8B or Mistral 7B via HF Inference API. Used only in the live Gradio demo. Stronger model = more impressive live behavior for judges. Not the model being trained — that distinction should be clearly noted in the README.

This separation means your training evidence (reward curve, loss curve, before/after traces) comes from the small model you actually trained, while the live demo uses API credits to show what a capable agent looks like in the environment. Both are honest — they serve different purposes.

---

## 10. Evaluation

### Baselines

| Agent | Expected Success Rate | Purpose |
|---|---|---|
| Random | ~10% | Floor — proves task is non-trivial |
| Heuristic (highest CPU = root cause) | ~22–30% | Ceiling without reasoning |
| Trained LLM agent | >65% | Target — must beat heuristic by >35% |

### Primary Metrics

| Metric | Definition |
|---|---|
| Task Success Rate | % episodes where health fully restored |
| Diagnosis Accuracy | % episodes where `diagnose()` is correct |
| Mean Steps to Resolution | Average steps on successful episodes |
| Reasoning Lift | Success rate delta between correct-diagnosis and no-diagnosis episodes |

**Reasoning Lift** is the key metric. It answers: does correct diagnosis actually help? If yes, the environment is functioning correctly and the agent is learning causal structure, not shortcuts.

### Secondary Metrics

| Metric | Definition |
|---|---|
| Cascade Rate | % episodes ending in system collapse |
| Tool Utilization | Ratio of informative to total actions |
| Reward per Episode | Total cumulative reward curve over training |

---

## 11. File Structure

```
incident-response-env/
├── env/
│   ├── environment.py        # OpenEnv interface: reset(), step(), render()
│   ├── simulator.py          # Hidden state, propagation matrix, failure logic
│   └── grader.yaml           # OpenEnv grader config
├── tools/
│   └── tools.py              # check_logs(), fix actions, is_fix_action()
├── agent/
│   └── model.py              # Random, heuristic, and LLM agent implementations
├── eval/
│   └── evaluate.py           # Evaluation harness, metric logging
├── train.py                  # TRL + Unsloth training loop
├── demo/
│   └── app.py                # Gradio Space demo
└── README.md                 # Problem, architecture, results, demo link
```

---

## 12. Tech Stack

| Tool | Role |
|---|---|
| OpenEnv (latest) | Environment framework and grader interface |
| HuggingFace TRL | RL training via GRPO |
| Unsloth | Efficient training on free-tier compute |
| HuggingFace Spaces + Gradio | Live demo and leaderboard |
| HuggingFace Hub | Model, dataset, and artifact publishing |
| Python 3.11+ | Primary language |
| PyTorch | Training backend |
| FastAPI | Environment server (optional, for Spaces) |
| Google Colab | Training notebook |
| GitHub | Version control |

---

## 13. Deliverables

| Deliverable | Description |
|---|---|
| Working environment | `reset()` + `step()` loop, fully runnable, seeded |
| Three agents | Random, heuristic, trained — all evaluated |
| Training script | `train.py` using TRL + Unsloth, runnable on free Colab |
| Reward curve | Shows improvement over training iterations |
| Loss curve | Policy loss over training |
| Before/after traces | Episode rollouts showing untrained vs trained behavior |
| Gradio Space | Live demo, accessible via URL, judge-runnable |
| HF Dataset | Published episode rollouts |
| Trained model | Qwen 1.5B on HF Hub with model card |
| README | Problem, design, results, links — mirrors problem statement language |

---

## 14. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Sparse reward early in training | Health delta from step 1 provides immediate signal |
| Reward hacking via fix spam | Wrong-service penalty + diminishing returns makes spam net-negative |
| `memory_leak` recurrence breaks training | Cap recurrence to once per episode in v1; add in v2 if stable |
| Free Colab disconnect during training | Checkpoint every 50 episodes; resume from last checkpoint |
| Environment too hard for 1.5B model | Start with 2 services + 2 failure modes; scale up after first successful training run |
| Token budget drained before finale ends | Semaphore limiting concurrent demo episodes; 15-step cap per demo; heuristic agent fallback ready |
| Demo model confused by environment format | Pre-test Llama 3.1 8B on 20 manual episodes before finale; confirm it can parse observation JSON and produce valid actions |

---

## 15. The Pitch

*Most agent benchmarks reward lucky outcomes the same as good reasoning. We built a stateful professional environment where an LLM agent must explicitly commit to a diagnosis before acting — making causal reasoning measurable, trainable, and structurally impossible to shortcut. The environment captures a partially observable world, requires belief updating from tool feedback, and produces a reward signal that reflects how well the agent actually understands the system — not just whether it got lucky.*

---

*Version 2.0 — Incident Response Agent / OpenEnv Hackathon*
