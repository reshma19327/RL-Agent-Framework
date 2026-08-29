from agents.q_learning import train_agent
from environments.gridworld import GridWorld


# Train the agent
print("Training Q-Learning agent...\n")

agent, rewards = train_agent(episodes=1000)


# Create a new environment for evaluation
env = GridWorld()

successful_episodes = 0
total_episodes = 100


# Evaluate the trained agent
for episode in range(total_episodes):

    state = env.reset()
    done = False

    for step in range(30):

        # Always choose the best learned action
        state_index = agent.state_to_index(state)
        action = agent.q_table[state_index].argmax()

        next_state, reward, done = env.step(action)

        state = next_state

        if done:
            successful_episodes += 1
            break


# Calculate success rate
success_rate = (successful_episodes / total_episodes) * 100


print("\n========== EVALUATION RESULTS ==========")
print(f"Evaluation Episodes : {total_episodes}")
print(f"Successful Episodes : {successful_episodes}")
print(f"Success Rate        : {success_rate:.2f}%")