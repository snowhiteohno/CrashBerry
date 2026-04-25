import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import IncidentResponseEnv

def trace_memory_leak():
    print("=== TRACING MEMORY LEAK SCENARIO ===")
    env = IncidentResponseEnv()
    
    # We will loop until we find a seed where root is 'api-gateway' and failure_mode is 'memory_leak'
    seed = 0
    while True:
        obs = env.reset(seed=seed)
        if env.simulator.root_failure_mode == 'memory_leak':
            break
        seed += 1
        
    root = env.simulator.root_cause_service
    fm = env.simulator.root_failure_mode
    print(f"Seed: {seed} | Root Cause: {root} | Mode: {fm}")
    print("--------------------------------------")
    
    # Step 1: Check logs
    action1 = {"type": "check_logs", "target": root}
    obs, r, d, i = env.step(action1)
    print(f"Step 1 | Action: {action1['type']} {root} | Reward: {r}")
    print(f"Observation diagnosis_made: {obs['diagnosis_made']}")
    
    # Step 2: Diagnose
    action2 = {"type": "diagnose", "target": root, "failure_mode": fm}
    obs, r, d, i = env.step(action2)
    print(f"Step 2 | Action: {action2['type']} | Reward: {r}")
    print(f"Observation diagnosis_made: {obs['diagnosis_made']}")
    
    # Step 3: Fix (restart_service handles memory_leak)
    action3 = {"type": "restart_service", "target": root}
    obs, r, d, i = env.step(action3)
    print(f"Step 3 | Action: {action3['type']} | Reward: {r}") # Should be +10 (+ health delta capped at 3)
    
    # We will now just no_op and watch the memory leak recur
    for step in range(4, 11):
        obs, r, d, i = env.step({"type": "no_op"})
        print(f"Step {step} | System Health: {obs['system_health_score']} | True Root Deg: {env.simulator.base_degradation[env.simulator.root_idx]} | Reward: {r}")

if __name__ == "__main__":
    trace_memory_leak()
