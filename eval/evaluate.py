"""Evaluation harness for the Incident Response environment.
Runs a given agent for a number of episodes, collects the metrics defined in
the PRD, and prints a summary. The script is deliberately lightweight – it
does not depend on any external logging framework to keep the repository
portable.
"""
import argparse
from typing import Dict, Any, List

from ..env.environment import IncidentResponseEnv
from ..agent.model import get_agent


def run_episode(env: IncidentResponseEnv, agent) -> Dict[str, Any]:
    """Run a single episode and return a dict with episode statistics.
    The returned dict contains:
    - success (bool)
    - diagnosis_correct (bool)
    - steps (int)
    - cumulative_reward (float)
    - termination_reason (str)
    """
    obs = env.reset()
    done = False
    cumulative_reward = 0.0
    diagnosis_correct = False
    steps = 0
    while not done:
        action = agent.select_action(obs)
        # For deterministic agents we may need to pass hidden info for check_logs
        if action.get("type") == "check_logs":
            # provide hidden root info for better hints (simulated)
            # In a real env this would be a method; here we just call the tool.
            from ..tools.tools import check_logs
            hints = check_logs(action["target"], env._sim.root_service, env._sim.root_failure)
            # Replace the action with a no_op and attach hints in observation
            # The environment does not process check_logs directly, so we just log.
            # This placeholder ensures the agent does not crash.
            action = {"type": "no_op"}
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward
        steps += 1
        # Detect correct diagnosis from info – the simulator updates
        # ``diagnosis_made`` flag but does not expose correctness. We infer from reward.
        if action.get("type") == "diagnose" and reward >= 8.0:
            diagnosis_correct = True
    # Determine termination reason
    termination_reason = "max_steps"
    if info.get("success_steps", 0) >= 2:
        termination_reason = "success"
    elif info.get("collapse_steps", 0) >= 3:
        termination_reason = "collapse"
    return {
        "success": termination_reason == "success",
        "diagnosis_correct": diagnosis_correct,
        "steps": steps,
        "cumulative_reward": cumulative_reward,
        "termination_reason": termination_reason,
    }


def evaluate(agent_name: str, num_episodes: int = 100, seed: int | None = None) -> List[Dict[str, Any]]:
    """Create the environment and agent, run ``num_episodes`` episodes, and
    return a list of episode result dictionaries.
    """
    env = IncidentResponseEnv(seed=seed)
    agent = get_agent(agent_name)
    results = []
    for _ in range(num_episodes):
        results.append(run_episode(env, agent))
    return results


def summarize(results: List[Dict[str, Any]]) -> None:
    total = len(results)
    success = sum(r["success"] for r in results)
    diag = sum(r["diagnosis_correct"] for r in results)
    avg_steps = sum(r["steps"] for r in results) / total
    avg_reward = sum(r["cumulative_reward"] for r in results) / total
    print("--- Evaluation Summary ---")
    print(f"Episodes: {total}")
    print(f"Success Rate: {success / total:.2%}")
    print(f"Diagnosis Accuracy: {diag / total:.2%}")
    print(f"Mean Steps to Resolution: {avg_steps:.2f}")
    print(f"Average Reward per Episode: {avg_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an agent on the incident response env")
    parser.add_argument("agent", choices=["random", "heuristic", "llm"], help="Agent to evaluate")
    parser.add_argument("-n", "--num-episodes", type=int, default=100, help="Number of episodes to run")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()
    results = evaluate(args.agent, args.num_episodes, args.seed)
    summarize(results)
