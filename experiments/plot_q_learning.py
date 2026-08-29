import matplotlib.pyplot as plt
from agents.q_learning import train_agent


# Train the agent
agent, rewards = train_agent(episodes=1000)


# Calculate moving average
window = 50
moving_average = []

for i in range(len(rewards)):
    start = max(0, i - window + 1)
    average = sum(rewards[start:i + 1]) / (i - start + 1)
    moving_average.append(average)


# Create graph
plt.figure(figsize=(10, 5))

plt.plot(rewards, alpha=0.3, label="Episode Reward")
plt.plot(
    moving_average,
    linewidth=2,
    label="Moving Average (50 episodes)"
)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning Training Performance")
plt.legend()
plt.grid()

# Save graph
plt.savefig("results/q_learning_convergence.png")

plt.show()