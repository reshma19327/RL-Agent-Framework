from environments.gridworld import GridWorld


env = GridWorld()

print("Initial Grid:")
env.render()

print("\nMoving Right...")

state, reward, done = env.step(3)

print("State:", state)
print("Reward:", reward)
print("Done:", done)

print("\nGrid after movement:")
env.render()