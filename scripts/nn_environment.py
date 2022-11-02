import gym
from src.generation import Generation
from src.utils import *
from src.genotypes import *
import src.policies as policies
import src.config as config
import src.evolution as evolution

env = gym.make("CartPole-v1")

generation_history = []
best_list = []
best_candidate = NeuralNetwork('0-0')
next_gen = None
target_fitness = config.data['evolution']['termination_fitness']

# code for "Live" plotting
plt.ion()
fig = plt.figure()


for i in range(config.data['evolution']['generations']):
    try:
        observation, _ = env.reset()

        generation = Generation('nn', i + 1, next_gen)

        for phenotype in generation.population:
            phenotype.find_phenotype_fitness(env, policies.nn_basic_encoding)

        generation.sort_population_by_fitness()
        generation_history.append(generation)  # Save the generation

        new_candidate = generation.population[-1]  # The best candidate of the generation
        best_list.append(new_candidate.fitness)

        if new_candidate.fitness > best_candidate.fitness:  # Update best candidate
            best_candidate = new_candidate

        next_gen = evolution.generate_offspring_nn(generation)  # Generate next generation

        print(f"Generation {1 + i}: Best individual fitness: {new_candidate.fitness}, "
              f"id: {new_candidate.candidate_number}")

        # Plot the best fitness for each iteration
        plt.plot(best_list, color='blue')
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.title(label='NN Evolution',
                  fontweight=10,
                  pad='2.0')

        fig.canvas.draw()
        fig.canvas.flush_events()

        if best_candidate.fitness > target_fitness:  # Termination condition
            break

    except KeyboardInterrupt:
        break


final_fitnesses = []
for _ in range(50):
    try:
        observation, _ = env.reset()

        best_candidate.find_phenotype_fitness(env, policies.nn_basic_encoding)
        final_fitnesses.append(best_candidate.fitness)

    except KeyboardInterrupt:
        break

average_fitness = np.mean(final_fitnesses)

save_nn_results(f"{generation_history[-1].population[-1].__str__()}\n"
                f"\n{final_fitnesses}\n"
                f"Average fitness over {len(final_fitnesses)} runs: {average_fitness}", fig)

plt.close()
env.close()
print(f"Done!")



