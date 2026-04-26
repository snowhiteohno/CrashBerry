import inspect
import sys

# --- SAFETY SHIELD: Monkey-patch PPOConfig before anything else imports it ---
try:
    import trl
    if hasattr(trl, "PPOConfig"):
        original_init = trl.PPOConfig.__init__
        def patched_init(self, *args, **kwargs):
            sig = inspect.signature(original_init).parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig or k == 'self'}
            return original_init(self, *args, **filtered_kwargs)
        trl.PPOConfig.__init__ = patched_init
except:
    pass

import os
import argparse
import json
import torch
import re

try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import trl
try:
    from trl import PPOTrainer, PPOConfig, create_reference_model
except ImportError:
    from trl.trainer import PPOTrainer, PPOConfig
    from trl.models import create_reference_model

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
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quantization_config, device_map=device, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

def parse_action(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return {"type": "no_op"}

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = IncidentResponseEnv(seed=args.seed)
    model, tokenizer = load_model(args.model_name, device)
    ref_model = create_reference_model(model)

    ppo_config = PPOConfig(
        batch_size=args.batch_size,
        mini_batch_size=args.batch_size,
        learning_rate=args.lr,
        log_with=None
    )

    trainer = PPOTrainer(model=model, ref_model=ref_model, tokenizer=tokenizer, config=ppo_config)

    print("🚀 Starting Training Loop...")
    for epoch in range(args.epochs):
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        while not done and step_count < env.max_steps:
            prompt = f"System: You are an incident response agent. Current State: {obs}\nAction (JSON format):"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            output = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7)
            response_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            action = parse_action(response_text)
            new_obs, reward, done, info = env.step(action)
            
            # Simple scalar reward
            trainer.step([inputs['input_ids'][0]], [output[0]], [torch.tensor(float(reward))])
            
            total_reward += reward
            obs = new_obs
            step_count += 1
            
        print(f"✅ Epoch {epoch+1} Complete | Total Reward: {total_reward:.2f}")

    print("🎉 Training process finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
