#
#  https://www.gymlibrary.dev/environments/classic_control/cart_pole/
#
#

import gym
import matplotlib.pyplot as plt
import policies
from generation import Generation
from utils.utils import *
from genotypes import CellularAutomaton1D
import evolution
import config

env = gym.make("CartPole-v1",)

generation_history = []
best_list = []
best_candidate = CellularAutomaton1D('0-0')
next_gen = None
plt.ion()
fig = plt.figure()
target_fitness = config.data['evolution']['termination_fitness']

for i in range(config.data['evolution']['generations']):

    observation, _ = env.reset()

    generation = Generation('ca', i + 1, next_gen)

    for phenotype in generation.population:
        phenotype.find_phenotype_fitness(env, policies.wide_encoding)
        #print(f"Phenotype fitness: {phenotype.get_fitness()}")
        #plt.imshow(phenotype.get_history(), cmap='gray')
        #plt.show()

    generation_history.append(generation)
    generation.sort_population_by_fitness()

    # print([g.get_fitness() for g in generation.population])
    # print([g.candidate_number for g in generation.population])

    fitnesses = np.asarray(generation.get_population_fitness())
    print(f"Generation {1+i}: Best individual fitness: {generation[-1].get_fitness()}, "
          f"rule: {binary_to_int(generation[-1].rule)} "
          f"id: {generation[-1].candidate_number}")

    new_candidate = generation.population[-1]

    if new_candidate.get_fitness() > best_candidate.get_fitness():
        best_candidate = new_candidate

    best_list.append(fitnesses[-1])

    plt.plot([g.get_population_fitness() for g in generation_history])

    fig.canvas.draw()
    fig.canvas.flush_events()

    if fitnesses[-1] > target_fitness:
        break

    next_gen = evolution.generate_offspring_ca(generation)

env2 = gym.make("CartPole-v1", render_mode="human" )
while True:

    max_steps = 500
    observation, _ = env2.reset()
    score = 0

    for i in range(max_steps):

        action = policies.wide_encoding(observation, best_candidate)  # User-defined policy function
        observation, reward, terminated, truncated, _ = env2.step(action)
        score += reward

        if terminated:
            break
        elif truncated:
            break

    print(score)

