import numpy as np
import matplotlib.pyplot as plt


# Load reward data
q_rewards = np.loadtxt(
    "results/q_learning_rewards.csv",
    delimiter=","
)

dqn_rewards = np.loadtxt(
    "results/dqn_rewards.csv",
    delimiter=","
)

reinforce_rewards = np.loadtxt(
    "results/reinforce_rewards.csv",
    delimiter=","
)


# Moving average function
def moving_average(data, window=50):

    result = []

    for i in range(len(data)):

        start = max(0, i - window + 1)

        result.append(
            np.mean(data[start:i + 1])
        )

    return result


# Calculate moving averages
q_average = moving_average(q_rewards)

dqn_average = moving_average(dqn_rewards)

reinforce_average = moving_average(
    reinforce_rewards
)


# Create comparison graph
plt.figure(figsize=(10, 6))

plt.plot(
    q_average,
    linewidth=2,
    label="Q-Learning"
)

plt.plot(
    dqn_average,
    linewidth=2,
    label="DQN"
)

plt.plot(
    reinforce_average,
    linewidth=2,
    label="REINFORCE"
)


plt.xlabel("Episode")

plt.ylabel(
    "Average Total Reward"
)

plt.title(
    "RL Algorithm Convergence Comparison"
)

plt.legend()

plt.grid()


# Save graph
plt.savefig(
    "results/algorithm_comparison.png"
)

plt.show()