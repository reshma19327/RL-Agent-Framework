import numpy as np
import matplotlib.pyplot as plt


# Load DQN rewards
rewards = np.loadtxt(
    "results/dqn_rewards.csv",
    delimiter=","
)


# Calculate moving average
window = 50

moving_average = []

for i in range(len(rewards)):

    start = max(0, i - window + 1)

    average = np.mean(
        rewards[start:i + 1]
    )

    moving_average.append(average)


# Create graph
plt.figure(figsize=(10, 5))

plt.plot(
    rewards,
    alpha=0.3,
    label="Episode Reward"
)

plt.plot(
    moving_average,
    linewidth=2,
    label="Moving Average (50 Episodes)"
)

plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.title("DQN Training Performance")

plt.legend()

plt.grid()


# Save graph
plt.savefig(
    "results/dqn_convergence.png"
)

plt.show()