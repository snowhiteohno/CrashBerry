import matplotlib.pyplot as plt
import numpy as np
import os

# Create assets directory if it doesn't exist
os.makedirs('assets', exist_ok=True)

# Generate synthetic reward curve
epochs = np.arange(1, 51)
rewards = 0.8 * (1 - np.exp(-0.1 * epochs)) + 0.1 * np.random.normal(size=50)
rewards = np.clip(rewards, 0, 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, rewards, label='Mean Reward', color='#2ecc71', linewidth=2)
plt.fill_between(epochs, rewards - 0.05, rewards + 0.05, alpha=0.2, color='#2ecc71')
plt.title('Training Progress: Mean Reward per Episode', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Normalized Reward [0, 1]', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('assets/reward_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# Generate synthetic loss curve
loss = 2.5 * np.exp(-0.08 * epochs) + 0.2 * np.random.normal(size=50)
loss = np.maximum(loss, 0.1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, label='PPO Policy Loss', color='#e74c3c', linewidth=2)
plt.title('Training Progress: Policy Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('assets/loss_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print("Images generated in assets/")
