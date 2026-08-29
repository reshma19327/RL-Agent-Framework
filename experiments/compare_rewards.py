import numpy as np

from agents.q_learning import QLearningAgent
from environments.gridworld import GridWorld


# ============================================================
# Train Q-Learning with a selected reward strategy
# ============================================================

def train_with_reward_strategy(
    strategy,
    episodes=1000
):

    env = GridWorld(
        reward_strategy=strategy
    )

    agent = QLearningAgent(
        state_size=25,
        action_size=4
    )

    rewards = []

    print(
        f"\nTraining with {strategy} reward..."
    )

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0

        done = False


        for step in range(100):

            # Choose action
            action = agent.choose_action(
                state
            )

            # Take action
            next_state, reward, done = env.step(
                action
            )

            # Update Q-table
            agent.learn(
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state

            total_reward += reward

            if done:
                break


        # Reduce exploration
        agent.decay_epsilon()

        rewards.append(
            total_reward
        )


        if (episode + 1) % 100 == 0:

            print(
                f"Episode {episode + 1}, "
                f"Reward: {total_reward}, "
                f"Epsilon: {agent.epsilon:.3f}"
            )


    return agent, rewards


# ============================================================
# Main experiment
# ============================================================

if __name__ == "__main__":

    strategies = [
        "sparse",
        "step",
        "shaped"
    ]


    all_results = {}


    for strategy in strategies:

        agent, rewards = train_with_reward_strategy(
            strategy,
            episodes=1000
        )

        all_results[strategy] = rewards


        # Save rewards
        np.savetxt(
            f"results/reward_{strategy}.csv",
            rewards,
            delimiter=","
        )


    print(
        "\n========================================"
    )

    print(
        "REWARD STRATEGY EXPERIMENT COMPLETED"
    )

    print(
        "========================================"
    )

    print(
        "Saved files:"
    )

    print(
        "results/reward_sparse.csv"
    )

    print(
        "results/reward_step.csv"
    )

    print(
        "results/reward_shaped.csv"
    )