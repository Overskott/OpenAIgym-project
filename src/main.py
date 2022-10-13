#
#  https://www.gymlibrary.dev/environments/classic_control/cart_pole/
#
#
import random

import gym
import numpy as np
import matplotlib.pyplot as plt
import policies
from genotypes import CellularAutomaton1D
import utils

env = gym.make("CartPole-v1")
results = [0, 0, 0]
seed = random.randint(0, 10000)

for run in range(258):
    score = 0
    rule = utils.int_to_binary(90)

    observation, _ = env.reset(seed=seed)

    for _ in range(500):

        configuration = np.zeros(50)
        model = CellularAutomaton1D(configuration, rule)

        action = policies.simple_ca(observation, model)  # User-defined policy function
        observation, reward, terminated, truncated, _ = env.step(action)

        score += reward

        if terminated:
            observation = env.reset()
            print(f"Run terminated with score {score}")
            break
        elif truncated:
            observation = env.reset()
            print(f"Run truncated with score {score}")
            break

    env.close()

    print(f"Run {run}, score {score}, rule {rule}")

    if score > results[1]:
        results[0] = run
        results[1] = score
        results[2] = rule

    plt.imshow(model.get_history(), cmap='gray')
    plt.show()

print(f"Best run was run #{results[0]} with the score {results[1]} using rule {results[2]} with seed {seed}")