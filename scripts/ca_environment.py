import gym
import src.policies as policies
from src.generation import Generation
from src.utils import *
from src.genotypes import CellularAutomaton1D
import src.evolution as evolution
import src.config as config


env = gym.make("CartPole-v1",)

generation_history = []
best_list = []
best_candidate = CellularAutomaton1D('0-0')
next_gen = None
plt.ion()
fig1 = plt.figure()
target_fitness = config.data['evolution']['termination_fitness']

for i in range(config.data['evolution']['generations']):

    try:
        observation, _ = env.reset()

        generation = Generation('ca', i + 1, next_gen)

        for phenotype in generation.population:
            phenotype.find_phenotype_fitness(env, policies.ca_wide_encoding)

            # if phenotype.get_fitness() > 200:
            #     plt.imshow(phenotype.get_history(), cmap='gray')
            #     plt.show()

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

        plt.plot(best_list)

        fig1.canvas.draw()
        fig1.canvas.flush_events()

        if fitnesses[-1] > target_fitness:
            break

        next_gen = evolution.generate_offspring_ca(generation)
    except KeyboardInterrupt:
        break


save_ca_results(generation_history[-1].population[-1].__str__())
save_figure(fig1, 'fitness')

fig2 = plt.figure()
plt.imshow(best_candidate.get_history(), cmap='gray')
save_figure(fig2, 'CA')

env2 = gym.make("CartPole-v1", render_mode="human")

average_score = 0
runs = 0
while True:
    try:
        max_steps = 500
        observation, _ = env2.reset()
        score = 0

        for i in range(max_steps):

            action = policies.ca_wide_encoding(observation, best_candidate)  # User-defined policy function
            observation, reward, terminated, truncated, _ = env2.step(action)
            score += reward

            if terminated:
                break
            elif truncated:
                break

        print(score)
        average_score += score
        runs += 1
    except KeyboardInterrupt:
        break




save_ca_results(average_score/runs)