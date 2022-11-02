import gym
from src import *

env2 = gym.make("CartPole-v1", render_mode="human")

candidate = generation.CellularAutomaton1D('0-0')

for _ in range(50):
    try:
        max_steps = 500
        observation, _ = env2.reset()
        score = 0

        for _ in range(max_steps):

            action = policies.ca_wide_encoding(observation, candidate)  # User-defined policy function
            observation, reward, terminated, truncated, _ = env2.step(action)
            score += reward

            if terminated:
                break
            elif truncated:
                break

        print(score)

    except KeyboardInterrupt:
        break