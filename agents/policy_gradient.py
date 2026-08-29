import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environments.gridworld import GridWorld


# ============================================================
# 1. Policy Network
# ============================================================

class PolicyNetwork(nn.Module):

    def __init__(self, state_size=25, action_size=4):

        super(PolicyNetwork, self).__init__()

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
# 2. REINFORCE Agent
# ============================================================

class REINFORCEAgent:

    def __init__(
        self,
        state_size=25,
        action_size=4,
        learning_rate=0.001,
        discount_factor=0.9
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Policy network
        self.policy_network = PolicyNetwork(
            state_size,
            action_size
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate
        )


    # ========================================================
    # Convert state to vector
    # ========================================================

    def state_to_vector(self, state):

        row, col = state

        state_index = row * 5 + col

        vector = np.zeros(
            self.state_size,
            dtype=np.float32
        )

        vector[state_index] = 1.0

        return vector


    # ========================================================
    # Choose Action
    # ========================================================

    def choose_action(self, state):

        state_vector = self.state_to_vector(state)

        state_tensor = torch.tensor(
            state_vector,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        # Get action probabilities
        logits = self.policy_network(
            state_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        # Sample action
        distribution = torch.distributions.Categorical(
            probabilities
        )

        action = distribution.sample()

        log_probability = distribution.log_prob(
            action
        )

        return action.item(), log_probability


    # ========================================================
    # Calculate Returns
    # ========================================================

    def calculate_returns(self, rewards):

        returns = []

        cumulative_reward = 0

        for reward in reversed(rewards):

            cumulative_reward = (
                reward
                + self.discount_factor * cumulative_reward
            )

            returns.insert(
                0,
                cumulative_reward
            )

        returns = torch.tensor(
            returns,
            dtype=torch.float32
        ).to(self.device)

        return returns


    # ========================================================
    # Update Policy
    # ========================================================

    def update_policy(
        self,
        log_probabilities,
        rewards
    ):

        returns = self.calculate_returns(
            rewards
        )

        loss = 0

        for log_probability, return_value in zip(
            log_probabilities,
            returns
        ):

            loss += -log_probability * return_value

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()


# ============================================================
# 3. Train REINFORCE
# ============================================================

def train_reinforce(episodes=1000):

    env = GridWorld()

    agent = REINFORCEAgent()

    rewards_history = []

    losses = []


    print("Training REINFORCE agent...\n")


    for episode in range(episodes):

        state = env.reset()

        episode_rewards = []

        log_probabilities = []

        done = False


        for step in range(100):

            action, log_probability = agent.choose_action(
                state
            )

            next_state, reward, done = env.step(
                action
            )

            log_probabilities.append(
                log_probability
            )

            episode_rewards.append(
                reward
            )

            state = next_state


            if done:
                break


        # Update policy
        loss = agent.update_policy(
            log_probabilities,
            episode_rewards
        )

        total_reward = sum(
            episode_rewards
        )

        rewards_history.append(
            total_reward
        )

        losses.append(
            loss
        )


        if (episode + 1) % 100 == 0:

            print(
                f"Episode {episode + 1}, "
                f"Reward: {total_reward}, "
                f"Loss: {loss:.4f}"
            )


    # Save rewards
    np.savetxt(
        "results/reinforce_rewards.csv",
        rewards_history,
        delimiter=","
    )


    # Save losses
    np.savetxt(
        "results/reinforce_losses.csv",
        losses,
        delimiter=","
    )


    print(
        "\nREINFORCE training completed!"
    )


    return agent, rewards_history, losses


# ============================================================
# 4. Evaluate REINFORCE
# ============================================================

def evaluate_reinforce(
    agent,
    episodes=100
):

    env = GridWorld()

    successful_episodes = 0

    total_steps = 0


    for episode in range(episodes):

        state = env.reset()

        done = False


        for step in range(30):

            # Get action probabilities
            state_vector = agent.state_to_vector(
                state
            )

            state_tensor = torch.tensor(
                state_vector,
                dtype=torch.float32
            ).unsqueeze(0).to(agent.device)


            with torch.no_grad():

                logits = agent.policy_network(
                    state_tensor
                )

                action = torch.argmax(
                    logits,
                    dim=1
                ).item()


            next_state, reward, done = env.step(
                action
            )

            state = next_state


            if done:

                successful_episodes += 1

                total_steps += step + 1

                break


    success_rate = (
        successful_episodes
        / episodes
    ) * 100


    if successful_episodes > 0:

        average_steps = (
            total_steps
            / successful_episodes
        )

    else:

        average_steps = 0


    print(
        "\n========== REINFORCE EVALUATION =========="
    )

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
    with open(
        "results/reinforce_evaluation.txt",
        "w"
    ) as file:

        file.write(
            "REINFORCE Evaluation Results\n"
        )

        file.write(
            "============================\n"
        )

        file.write(
            f"Evaluation Episodes: {episodes}\n"
        )

        file.write(
            f"Successful Episodes: {successful_episodes}\n"
        )

        file.write(
            f"Success Rate: {success_rate:.2f}%\n"
        )

        file.write(
            f"Average Steps: {average_steps:.2f}\n"
        )


    return success_rate, average_steps


# ============================================================
# 5. Main Program
# ============================================================

if __name__ == "__main__":

    agent, rewards, losses = train_reinforce(
        episodes=1000
    )

    evaluate_reinforce(
        agent,
        episodes=100
    )