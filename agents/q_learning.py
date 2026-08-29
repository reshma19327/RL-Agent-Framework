import numpy as np
from environments.gridworld import GridWorld


class QLearningAgent:

    def __init__(
        self,
        state_size=25,
        action_size=4,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table
        self.q_table = np.zeros((state_size, action_size))

    def state_to_index(self, state):
        """Convert (row, column) into a single state number."""
        row, col = state
        return row * 5 + col

    def choose_action(self, state):
        """Choose an action using epsilon-greedy strategy."""

        state_index = self.state_to_index(state)

        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        # Exploitation
        return np.argmax(self.q_table[state_index])

    def learn(self, state, action, reward, next_state, done):
        """Update the Q-table."""

        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)

        current_q = self.q_table[state_index, action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(
                self.q_table[next_state_index]
            )

        # Q-Learning update
        self.q_table[state_index, action] += (
            self.learning_rate * (target_q - current_q)
        )

    def decay_epsilon(self):
        """Reduce exploration over time."""

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


def train_agent(episodes=1000):

    env = GridWorld()
    agent = QLearningAgent()

    rewards = []

    for episode in range(episodes):

        state = env.reset()
        total_reward = 0
        done = False

        for step in range(100):

            action = agent.choose_action(state)

            next_state, reward, done = env.step(action)

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

        agent.decay_epsilon()

        rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1}, "
                f"Reward: {total_reward}, "
                f"Epsilon: {agent.epsilon:.3f}"
            )
        # Save rewards
    np.savetxt(
        "results/q_learning_rewards.csv",
        rewards,
        delimiter=","
    )

    return agent, rewards


if __name__ == "__main__":
    agent, rewards = train_agent()

    print("\nTraining completed!")