# Reinforcement Learning Agent Framework

## Student Details

**Full Name:** RESHMA AJESH THOMAS  
**Registered Email:** reshmaajeshthomas.b23cs2154@mbcet.ac.in

## Project Topic

Reinforcement Learning Agent Framework using Q-Learning, Deep Q Networks (DQN), and REINFORCE Policy Optimization.

## Project Overview

This project demonstrates autonomous learning in a GridWorld environment using three reinforcement learning algorithms:

- Q-Learning
- Deep Q Network (DQN)
- REINFORCE Policy Gradient

The project also compares three reward optimization strategies:

- Sparse Reward
- Step Penalty
- Reward Shaping

## Technologies Used

- Python
- NumPy
- Matplotlib
- PyTorch

## Results

| Algorithm | Success Rate | Avg. Steps |
|-----------|-------------:|-----------:|
| Q-Learning | 100% | 8 |
| DQN | 100% | 8 |
| REINFORCE | 100% | 8 |

## How to Run

```bash
pip install -r requirements.txt
python -m agents.q_learning
python -m agents.dqn
python -m agents.policy_gradient
```