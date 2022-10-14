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
size = 30
rule = np.random.randint(0, 2, 2**5, dtype='i1')

result_ca = CellularAutomaton1D(rule=rule, size=size, hood_size=5)

for run in range(500):
    score = 0
    size = 30
    rule = np.random.randint(0, 2, 2**5, dtype='i1')


    observation, _ = env.reset(seed=seed)

    for _ in range(500):

        model = CellularAutomaton1D(rule=rule, size=size, hood_size=5)

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
        result_ca = model


print(f"Best run was run #{results[0]} with the score {results[1]} using rule {utils.binary_to_int(results[2])} with seed {seed}")

plt.imshow(result_ca.get_history(), cmap='gray')
plt.show()
