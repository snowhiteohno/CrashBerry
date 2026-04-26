import os
import argparse
import json
import torch
import torch.nn.functional as F
import re
from tqdm import tqdm

# We ONLY use unsloth for the fast model, NO more TRL trainer to avoid conflicts
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from env.environment import IncidentResponseEnv

def load_model(model_name: str, device: str):
    if HAS_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
        )
        return model, tokenizer
    else:
        # Fallback to standard 4-bit loading
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

def parse_action(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: return json.loads(match.group(0))
    except: pass
    return {"type": "no_op"}

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = IncidentResponseEnv(seed=args.seed)
    model, tokenizer = load_model(args.model_name, device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    print("🚀 Starting Manual RL Training Loop (Policy Gradient)...")
    
    reward_history = []
    for epoch in range(args.epochs):
        obs = env.reset()
        done = False
        step_count = 0
        epoch_rewards = []
        
        # Track log-probabilities for the Policy Gradient update
        saved_log_probs = []
        rewards = []

        while not done and step_count < env.max_steps:
            prompt = f"System: Incident Response Agent. State: {obs}\nAction (JSON):"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Generate actual action text
            with torch.no_grad():
                gen_out = model.generate(**inputs, max_new_tokens=32, do_sample=True, temperature=0.7)
                response_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
            
            action = parse_action(response_text)
            new_obs, reward, done, info = env.step(action)
            
            # Policy Gradient Step (Simplified: Use reward to scale loss)
            # This is a basic REINFORCE-style update
            optimizer.zero_grad()
            (loss * -reward).backward() # Higher reward = lower negative loss = stronger reinforcement
            optimizer.step()
            
            epoch_rewards.append(reward)
            obs = new_obs
            step_count += 1
            
        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
        reward_history.append(avg_reward)
        print(f"✅ Epoch {epoch+1}/{args.epochs} | Avg Reward: {avg_reward:.4f} | Steps: {step_count}")

    print("🎉 Training Finished! Plotting results...")
    import matplotlib.pyplot as plt
    plt.plot(reward_history)
    plt.title("Reward Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Reward")
    plt.savefig("assets/reward_curve_new.png")
    print("New reward curve saved to assets/reward_curve_new.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
