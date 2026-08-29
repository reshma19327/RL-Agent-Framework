import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environments.gridworld import GridWorld


# ============================================================
# 1. DQN Neural Network
# ============================================================

class DQN(nn.Module):

    def __init__(self, state_size=25, action_size=4):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, action_size)
        )

    def forward(self, state):
        return self.network(state)


# ============================================================
# 2. Experience Replay Memory
# ============================================================

class ReplayMemory:

    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ============================================================
# 3. DQN Agent
# ============================================================

class DQNAgent:

    def __init__(
        self,
        state_size=25,
        action_size=4,
        learning_rate=0.001,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=32
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Main network
        self.policy_network = DQN(
            state_size,
            action_size
        ).to(self.device)

        # Target network
        self.target_network = DQN(
            state_size,
            action_size
        ).to(self.device)

        # Copy initial weights
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate
        )

        # Loss function
        self.loss_function = nn.MSELoss()

        # Replay memory
        self.memory = ReplayMemory(capacity=10000)


    # ========================================================
    # Convert GridWorld state to neural network input
    # ========================================================

    def state_to_vector(self, state):

        row, col = state

        state_index = row * 5 + col

        vector = np.zeros(self.state_size, dtype=np.float32)

        vector[state_index] = 1.0

        return vector


    # ========================================================
    # Choose Action using Epsilon-Greedy
    # ========================================================

    def choose_action(self, state):

        # Exploration
        if random.random() < self.epsilon:

            return random.randrange(self.action_size)

        # Exploitation
        state_vector = self.state_to_vector(state)

        state_tensor = torch.tensor(
            state_vector,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            q_values = self.policy_network(
                state_tensor
            )

        return torch.argmax(q_values).item()


    # ========================================================
    # Store Experience
    # ========================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        state_vector = self.state_to_vector(state)

        next_state_vector = self.state_to_vector(
            next_state
        )

        self.memory.push(
            state_vector,
            action,
            reward,
            next_state_vector,
            done
        )


    # ========================================================
    # Train from Replay Memory
    # ========================================================

    def replay(self):

        if len(self.memory) < self.batch_size:

            return None

        batch = self.memory.sample(
            self.batch_size
        )

        states = torch.tensor(
            np.array([experience[0] for experience in batch]),
            dtype=torch.float32
        ).to(self.device)

        actions = torch.tensor(
            [experience[1] for experience in batch],
            dtype=torch.long
        ).to(self.device)

        rewards = torch.tensor(
            [experience[2] for experience in batch],
            dtype=torch.float32
        ).to(self.device)

        next_states = torch.tensor(
            np.array([experience[3] for experience in batch]),
            dtype=torch.float32
        ).to(self.device)

        dones = torch.tensor(
            [experience[4] for experience in batch],
            dtype=torch.float32
        ).to(self.device)


        # Current Q values
        current_q_values = self.policy_network(
            states
        ).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)


        # Next Q values
        with torch.no_grad():

            next_q_values = self.target_network(
                next_states
            ).max(1)[0]


        # Bellman equation
        target_q_values = rewards + (
            1 - dones
        ) * self.discount_factor * next_q_values


        # Calculate loss
        loss = self.loss_function(
            current_q_values,
            target_q_values
        )


        # Backpropagation
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


        return loss.item()


    # ========================================================
    # Reduce Exploration
    # ========================================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


    # ========================================================
    # Update Target Network
    # ========================================================

    def update_target_network(self):

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )


# ============================================================
# 4. Train DQN
# ============================================================

def train_dqn(episodes=1000):

    env = GridWorld()

    agent = DQNAgent()

    rewards = []

    losses = []


    print("Training DQN agent...\n")

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0

        episode_losses = []

        done = False


        for step in range(100):

            # Choose action
            action = agent.choose_action(
                state
            )


            # Environment response
            next_state, reward, done = env.step(
                action
            )


            # Store experience
            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )


            # Train network
            loss = agent.replay()

            if loss is not None:

                episode_losses.append(loss)


            state = next_state

            total_reward += reward


            if done:

                break


        # Reduce exploration
        agent.decay_epsilon()


        # Update target network every 10 episodes
        if (episode + 1) % 10 == 0:

            agent.update_target_network()


        # Save results
        rewards.append(total_reward)


        if episode_losses:

            losses.append(
                np.mean(episode_losses)
            )

        else:

            losses.append(0)


        # Print progress
        if (episode + 1) % 100 == 0:

            print(
                f"Episode {episode + 1}, "
                f"Reward: {total_reward}, "
                f"Epsilon: {agent.epsilon:.3f}, "
                f"Loss: {losses[-1]:.4f}"
            )


    # Save rewards
    np.savetxt(
        "results/dqn_rewards.csv",
        rewards,
        delimiter=","
    )


    # Save losses
    np.savetxt(
        "results/dqn_losses.csv",
        losses,
        delimiter=","
    )


    print("\nDQN training completed!")

    return agent, rewards, losses


# ============================================================
# 5. Evaluate Trained DQN Agent
# ============================================================

def evaluate_dqn(agent, episodes=100):

    env = GridWorld()

    successful_episodes = 0

    total_steps = 0


    # Turn off exploration
    agent.epsilon = 0


    for episode in range(episodes):

        state = env.reset()

        done = False


        for step in range(30):

            action = agent.choose_action(
                state
            )

            next_state, reward, done = env.step(
                action
            )

            state = next_state


            if done:

                successful_episodes += 1

                total_steps += step + 1

                break


    success_rate = (
        successful_episodes / episodes
    ) * 100


    if successful_episodes > 0:

        average_steps = (
            total_steps / successful_episodes
        )

    else:

        average_steps = 0


    print("\n========== DQN EVALUATION ==========")

    print(
        f"Evaluation Episodes : {episodes}"
    )

    print(
        f"Successful Episodes : {successful_episodes}"
    )

    print(
        f"Success Rate        : {success_rate:.2f}%"
    )

    print(
        f"Average Steps       : {average_steps:.2f}"
    )
    # Save evaluation results
    with open("results/dqn_evaluation.txt", "w") as file:
        file.write("DQN Evaluation Results\n")
        file.write("======================\n")
        file.write(f"Evaluation Episodes: {episodes}\n")
        file.write(f"Successful Episodes: {successful_episodes}\n")
        file.write(f"Success Rate: {success_rate:.2f}%\n")
        file.write(f"Average Steps: {average_steps:.2f}\n")

    return success_rate, average_steps


# ============================================================
# 6. Main Program
# ============================================================

if __name__ == "__main__":

    # Train
    agent, rewards, losses = train_dqn(
        episodes=1000
    )


    # Evaluate
    evaluate_dqn(
        agent,
        episodes=100
    )