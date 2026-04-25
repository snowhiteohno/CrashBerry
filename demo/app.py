"""Gradio demo for the Incident Response environment.
The demo limits concurrent episodes to one via a semaphore and caps each
episode at 15 steps as required by the PRD. It displays the sequence of
observations and actions, the final reward, and whether the episode ended
in success or collapse.
"""
import threading
import os
import sys
import json
from typing import List, Dict, Any

# Add project root to the FRONT of sys.path so local modules always win.
# Also add common Colab paths as a fallback.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _candidate in [_PROJECT_ROOT, "/content/CrashBerry", os.getcwd()]:
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import gradio as gr

from env.environment import IncidentResponseEnv
from agent.model import get_agent

# Semaphore to ensure only one episode runs at a time.
_semaphore = threading.Semaphore(1)

# Environment configuration – we use a reduced max_steps of 15 for the demo.
_DEMO_MAX_STEPS = 15

def _run_demo_episode(agent_name: str) -> str:
    # Acquire semaphore; if unavailable return a busy message.
    if not _semaphore.acquire(blocking=False):
        return "Demo is busy — another episode is running. Try again later."
    try:
        env = IncidentResponseEnv()
        env.max_steps = _DEMO_MAX_STEPS  # enforce 15‑step cap
        try:
            agent = get_agent(agent_name)
        except Exception as e:
            return f"Error creating agent: {str(e)}"
        
        obs = env.reset()
        steps: List[Dict[str, Any]] = []
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(obs)
            # The LLMAgent may need HF credentials; we assume they are set in env vars.
            # For demo purposes, unknown actions fallback to no_op inside the agent.
            new_obs, reward, done, info = env.step(action)
            steps.append({"observation": obs, "action": action, "reward": reward})
            total_reward += reward
            obs = new_obs
            
        # Determine termination reason.
        termination = "max_steps"
        if info.get("success_steps", 0) >= 2:
            termination = "success"
        elif info.get("collapse_steps", 0) >= 3:
            termination = "collapse"
            
        result = {
            "steps": steps,
            "total_reward": total_reward,
            "termination": termination,
        }
        return _format_result(result)
    finally:
        _semaphore.release()

def _format_result(result: Dict[str, Any]) -> str:
    if "error" in result:
        return result["error"]
    lines = []
    for i, step in enumerate(result["steps"], start=1):
        obs = json.dumps(step["observation"], indent=2)
        act = json.dumps(step["action"], indent=2)
        lines.append(f"Step {i}:\nAction: {act}\nReward: {step['reward']:.2f}\nObservation: {obs}\n")
    lines.append(f"\nTotal Reward: {result['total_reward']:.2f}\nTermination: {result['termination']}")
    return "\n".join(lines)

with gr.Blocks() as demo:
    gr.Markdown("# Incident Response Demo\nChoose an agent and run a single episode (max 15 steps). Only one episode can run at a time.")
    agent_dropdown = gr.Dropdown(choices=["random", "heuristic", "llm"], label="Agent", value="random")
    run_button = gr.Button("Run Episode")
    output_box = gr.Textbox(label="Result", lines=30)
    run_button.click(fn=_run_demo_episode, inputs=[agent_dropdown], outputs=[output_box], api_name="run_episode")

if __name__ == "__main__":
    # Gradio dev server – the user can launch it with `python demo/app.py`
    demo.launch()
