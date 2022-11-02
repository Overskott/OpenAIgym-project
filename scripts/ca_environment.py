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
target_fitness = config.data['evolution']['termination_fitness']

# code for "Live" plotting
plt.ion()
fig1 = plt.figure()

for i in range(config.data['evolution']['generations']):

    try:  # Handling user exit of the loop
        observation, _ = env.reset()

        generation = Generation('ca', i + 1, next_gen)

        for phenotype in generation.population: # Evaluate fitness of each phenotype in the generation
            phenotype.find_phenotype_fitness(env, policies.ca_wide_encoding)

        generation.sort_population_by_fitness()
        generation_history.append(generation)  # Save the generation

        new_candidate = generation.population[-1]  # The best candidate of the generation
        best_list.append(new_candidate.fitness)

        if new_candidate.fitness > best_candidate.fitness: # Update best candidate
            best_candidate = new_candidate

        next_gen = evolution.generate_offspring_ca(generation)  # Generate next generation

        print(f"Generation {1 + i}: Best individual fitness: {new_candidate.fitness}, "
              f"rule: {binary_to_int(new_candidate.rule)} "
              f"id: {new_candidate.candidate_number}")

        # Plot the best fitness for each iteration
        # plt.plot([candidate.get_fitness() for candidate in best_list])
        plt.plot(best_list)
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.title(label='CA Evolution',
                  fontweight=10,
                  pad='2.0')

        fig1.canvas.draw()
        fig1.canvas.flush_events()

        if best_candidate.fitness > target_fitness:  # Termination condition
            break

    except KeyboardInterrupt:
        break

final_fitnesses = []

for _ in range(50):
    try:
        observation, _ = env.reset()

        best_candidate.find_phenotype_fitness(env, policies.ca_wide_encoding)
        final_fitnesses.append(best_candidate.fitness)
    except KeyboardInterrupt:
        break

average_fitness = np.mean(final_fitnesses)

save_ca_results(f"{generation_history[-1].population[-1].__str__()}\n"
                f"\n{final_fitnesses}\n"
                f"Average fitness over {len(final_fitnesses)} runs: {average_fitness}")
save_figure(fig1, 'fitness')

fig2 = plt.figure()
plt.imshow(best_candidate.get_history(), cmap='gray')
save_figure(fig2, 'CA')

plt.close()
env.close()
print(f"Done!")





