#
#  https://www.gymlibrary.dev/environments/classic_control/cart_pole/
#
#

import policies
from generation import Generation
from utils import *
from genotypes import *
from policies import *
import config
import evolution

env = gym.make("CartPole-v1")

generation_history = []
best_list = []
best_candidate = NeuralNetwork('0-0')
next_gen = None
target_fitness = config.data['evolution']['termination_fitness']

plt.ion()
fig = plt.figure()


for i in range(config.data['evolution']['generations']):

    observation, _ = env.reset()

    generation = Generation('nn', i + 1, next_gen)

    for phenotype in generation.population:
        phenotype.find_phenotype_fitness(env, policies.nn_basic_encoding)
        print(f"Fitness: {phenotype.get_fitness()}, "
              f"id: {phenotype.candidate_number}")

    generation_history.append(generation)
    generation.sort_population_by_fitness()
    print([g.get_fitness() for g in generation.population])
    print([g.candidate_number for g in generation.population])

    new_candidate = generation.population[-1]

    if new_candidate.get_fitness() > best_candidate.get_fitness():
        best_candidate = new_candidate


    fitnesses = np.asarray(generation.get_population_fitness())
    print(f"Generation {i}: Best individual fitness: {generation[-1].get_fitness()}, "
          f"id: {generation[-1].candidate_number}")

    best_list.append(fitnesses[-1])
    fitness_list = [g.get_population_fitness() for g in generation_history]

    plt.plot(best_list)
    #plt.scatter(range(len(fitness_list)), fitness_list)

    fig.canvas.draw()
    fig.canvas.flush_events()
    if fitnesses[-1] >= target_fitness:
        break
    next_gen = evolution.generate_offspring_nn(generation)

#write_to_file(generation_history[-1].population[-1].__str__())
save_nn_results(generation_history[-1].population[-1].__str__(), fig)

#plt.close(fig)
env2 = gym.make("CartPole-v1", render_mode="human" )
while True:

    max_steps = 500
    observation, _ = env2.reset()
    score = 0

    for i in range(max_steps):

        action = nn_basic_encoding(observation, best_candidate)  # User-defined policy function
        observation, reward, terminated, truncated, _ = env2.step(action)
        score += reward

        if terminated:
            break
        elif truncated:
            break

    print(score)

# environment.close() i in g.population] for g in generations])



