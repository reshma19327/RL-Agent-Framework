from agents.q_learning import train_agent
from environments.gridworld import GridWorld


# Train the agent
agent, rewards = train_agent(episodes=1000)

# Create environment
env = GridWorld()

# Start from beginning
state = env.reset()

print("\n========== LEARNED PATH ==========")
env.render()

done = False
step = 0

while not done and step < 30:

    # Choose best learned action
    state_index = agent.state_to_index(state)
    action = agent.q_table[state_index].argmax()

    # Take action
    next_state, reward, done = env.step(action)

    state = next_state
    step += 1

    print(f"\nStep {step} | Reward: {reward}")
    env.render()

if done:
    print("\n🎯 GOAL REACHED!")
    print(f"Total steps: {step}")
else:
    print("\n❌ Goal was not reached.")