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
import config

env = gym.make("CartPole-v1")
results = [0, 0, 0]

seed = 42
#np.random.seed(seed)
results = []
parents =[]
for i in range(config.data['evolution_steps']):

    observation, _ = env.reset()

    generation = Population(i)
    generation.parents = parents
    generation.get_next_generation()

    for phenotype in generation.population:
        phenotype.test_phenotype(env, policies.wide_encoding)
        #plt.imshow(phenotype.get_history(), cmap='gray')
        #plt.show()

    generation.sort_children_by_fitness()
    fitnesses = [p.get_fitness() for p in generation.population]
    result = sum(fitnesses[-4:]) / 4

    results.append(result)
    print(f"Generation {i}, best performer {fitnesses[-1]} average score: {result}")


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


