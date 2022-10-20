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
from utils import utils
import environment
from generation import Population

env = gym.make("CartPole-v1")
results = [0, 0, 0]

seed = 42
#np.random.seed(seed)

from utils.config_parser import get_config_file

data = get_config_file()['parameters']

generation = Population()
generation.initialize_population()
results = []

for i in range(data['evolution']['evolution_steps']):
    observation, _ = env.reset()

    for phenotype in generation.population:
        environment.test_genotype(env, phenotype, policies.wide_encoding)
        #plt.imshow(phenotype.get_history(), cmap = 'gray')
        #plt.show()
    print([x.get_fitness() for x in generation.population])

    parents = generation.select_parents()
    generation.create_next_generation(parents)

    result = sum([parent.get_fitness() for parent in parents])/len(parents)

    results.append(result)
    print(f"Generation {i}, average score: {result}")



print([g.get_fitness() for g in generation.population])

plt.plot(results)
plt.show()
# for run in range(10):
#     score = 0
#     rule = np.random.randint(0, 2, 2**5, dtype='i1')
#
#     observation, _ = env.reset(seed=seed)
#
#     for _ in range(500):
#
#         model = CellularAutomaton1D(rule=rule)
#
#         action = policies.simple_ca(observation, model)  # User-defined policy function
#         observation, reward, terminated, truncated, _ = env.step(action)
#
#         score += reward
#
#         if terminated:
#             observation = env.reset()
#             print(f"Run terminated with score {score}")
#             break
#         elif truncated:
#             observation = env.reset()
#             print(f"Run truncated with score {score}")
#             break
#
#     env.close()
#
#     print(f"Run {run}, score {score}, rule {rule}")
#
#     if score > results[1]:
#         results[0] = run
#         results[1] = score
#         results[2] = rule
#         result_ca = model
#
#
# print(f"Best run was run #{results[0]} with the score {results[1]} using rule {utils.binary_to_int(results[2])} with seed {seed}")


