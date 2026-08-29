import numpy as np
import matplotlib.pyplot as plt


# Load reward data
sparse = np.loadtxt(
    "results/reward_sparse.csv",
    delimiter=","
)

step = np.loadtxt(
    "results/reward_step.csv",
    delimiter=","
)

shaped = np.loadtxt(
    "results/reward_shaped.csv",
    delimiter=","
)


# Moving average
def moving_average(data, window=50):

    result = []

    for i in range(len(data)):

        start = max(0, i - window + 1)

        result.append(
            np.mean(data[start:i + 1])
        )

    return result


# Calculate moving averages
sparse_avg = moving_average(sparse)

step_avg = moving_average(step)

shaped_avg = moving_average(shaped)


# Plot
plt.figure(figsize=(10, 6))

plt.plot(
    sparse_avg,
    linewidth=2,
    label="Sparse Reward"
)

plt.plot(
    step_avg,
    linewidth=2,
    label="Step Penalty"
)

plt.plot(
    shaped_avg,
    linewidth=2,
    label="Reward Shaping"
)


plt.xlabel("Episode")

plt.ylabel(
    "Average Reward"
)

plt.title(
    "Reward Strategy Comparison"
)

plt.legend()

plt.grid()


# Save
plt.savefig(
    "results/reward_strategy_comparison.png"
)

plt.show()