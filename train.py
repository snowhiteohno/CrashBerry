try:
    import unsloth
    from unsloth import apply_chat_template, apply_unsloth
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

import os
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# TRL imports
import trl
if hasattr(trl, "PPOTrainer"):
    from trl import PPOTrainer, PPOConfig
elif hasattr(trl, "trainer") and hasattr(trl.trainer, "PPOTrainer"):
    from trl.trainer import PPOTrainer, PPOConfig
else:
    try:
        from trl.trainer.ppo_trainer import PPOTrainer
        from trl.trainer.ppo_config import PPOConfig
    except ImportError:
        PPOTrainer = None
        PPOConfig = None

try:
    from trl import create_reference_model
except ImportError:
    try:
        from trl.models import create_reference_model
    except ImportError:
        create_reference_model = None

# Import the environment.
from env.environment import IncidentResponseEnv


def load_model(model_name: str, device: str):
    """Load model with optional Unsloth optimizations."""
    if not HAS_UNSLOTH:
        print("⚠️ Unsloth not found. Falling back to standard Transformers (slower).")

    # Load base model. Standard HF loading.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Apply Unsloth optimizations only if available.
    if HAS_UNSLOTH:
        apply_unsloth(model)
    
    return model, tokenizer


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = IncidentResponseEnv(seed=args.seed)

    # Load model.
    model, tokenizer = load_model(args.model_name, device)
    # Create reference model for PPO (copy of the same weights, no gradient).
    ref_model = create_reference_model(model)

    # PPO configuration – values are chosen to be reasonable for a 1.5B model.
    ppo_config = PPOConfig(
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        learning_rate=args.lr,
        max_grad_norm=1.0,
        kl_coef=0.2,
        clip_range=0.2,
        clip_range_value=0.2,
        target_kl=None,
    )

    trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=ppo_config,
        dataset=None,  # We'll generate on‑the‑fly using the env.
    )

    # Simple training loop: generate an observation, have the model propose an
    # action (as a JSON string), parse it, step the env, and use the reward as
    # the PPO reward signal.
    reward_history = []
    for epoch in range(args.epochs):
        obs = env.reset()
        done = False
        while not done:
            # Serialize observation to a prompt.
            prompt = f"Observation:\n{tokenizer.encode(str(obs), add_special_tokens=False)}"
            # Let the model generate a JSON action.
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            generated = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
            output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            # Extract JSON action – same heuristic as in LLMAgent.
            try:
                start = output_text.find('{')
                end = output_text.rfind('}')
                action = eval(output_text[start : end + 1])  # unsafe but placeholder for demo
            except Exception:
                # fallback no_op on parsing failure
                action = {"type": "no_op"}
            # Step environment.
            new_obs, reward, done, info = env.step(action)
            # PPO step – we treat the reward as a scalar value.
            # The response is the generated text; the reference is the same text.
            response = generated.squeeze(0)
            # Compute logprobs for PPO (trainer expects tensors).
            scores = torch.tensor([reward], device=device)
            trainer.step([inputs['input_ids'], response], scores)
            reward_history.append(reward)
            obs = new_obs
        # Log after each epoch.
        avg_reward = sum(reward_history[-env.max_steps:]) / env.max_steps
        print(f"Epoch {epoch+1}/{args.epochs} – Avg reward per episode: {avg_reward:.2f}")
        # Save checkpoint.
        ckpt_dir = Path(args.output_dir) / f"checkpoint_epoch_{epoch+1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    # Save final model.
    final_dir = Path(args.output_dir) / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Training complete. Model saved to", final_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLM agent on Incident Response env using TRL + Unsloth")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen1.5-1.8B", help="HuggingFace model identifier for Qwen 1.5B variant")
    parser.add_argument("--output-dir", type=str, default="./trained_models", help="Directory to store checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (episodes) to run")
    parser.add_argument("--batch-size", type=int, default=1, help="PPO batch size (episodes are processed one‑by‑one)")
    parser.add_argument("--mini-batch-size", type=int, default=1, help="PPO mini‑batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for PPO")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)
