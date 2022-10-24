#
#  https://www.gymlibrary.dev/environments/classic_control/cart_pole/
#
#

import gym
import matplotlib.pyplot as plt
import policies
from generation import Generation
from utils.utils import *
import evolution
import config

env = gym.make("CartPole-v1")
results = [0, 0, 0]

seed = 42
#np.random.seed(seed)
generation_history = []
best_list = []
next_gen = None
plt.ion()
fig = plt.figure()


for i in range(config.data['evolution']['generations']):

    observation, _ = env.reset(seed=seed)

    generation = Generation(i + 1, next_gen)

    for phenotype in generation.population:
        phenotype.find_phenotype_fitness(env, policies.wide_encoding)
        #print(f"Phenotype fitness: {phenotype.get_fitness()}")
        #plt.imshow(phenotype.get_history(), cmap='gray')
        #plt.show()

    generation_history.append(generation)
    generation.sort_population_by_fitness()

    fitnesses = np.asarray(generation.get_population_fitness())
    print(f"Generation {i}: Best individual fitness: {generation[-1].get_fitness()}, "
          f"size: {generation[-1].size}, steps: {generation[-1].steps}, "
          f"rule: {binary_to_int(generation.population[-1].rule)}, "
          f"id: {id(generation[-1])}")

    best_list.append(fitnesses[-1])

    plt.plot([g.get_population_fitness() for g in generation_history])

    fig.canvas.draw()
    fig.canvas.flush_events()

    next_gen = evolution.generate_offspring_ca(generation)
    #print([binary_to_int(i.rule) for i in next_gen])

#print([[i.get_fitness() for i in g.population] for g in generations])

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


