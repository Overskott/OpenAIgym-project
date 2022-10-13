import random
import numpy as np
import policies
import gym

from genotypes import Genotype, CellularAutomaton1D


def binary_to_int(array: np.ndarray):
    """Converting a numpy array with binary values to decimal integer"""
    binary_string = ''.join(f"{binary}" for binary in array)
    number = int(binary_string, 2)

    return number


def observables_to_binary(observables):
    """Observables = [position, velocity, angle, angular_momentum]"""
    state = np.zeros(len(observables))
    for i, observable in enumerate(observables):
        if observable >= 0:
            state[i] = 1
        else:
            state[i] = 0
    return state


def init_generation(pop, gen, selection):

    for generation in range(gen):

        generation = []
        for individual in range(pop):

            size = np.random.randint(10, 101)
            hood_size = np.random.randint(3, 4)
            rule = np.random.randint(0, 2**hood_size)

            configuration = np.random.randint(0, 2, size, dtype='i1')
            individual = CellularAutomaton1D(configuration, rule, hood_size)

            individual.run_time_evolution(size)

            generation.append(individual)


def run_pole_cart(model: CellularAutomaton1D):

    env = gym.make("CartPole-v1")
    results = [0, 0, 0]
    seed = random.randint(0, 10000)
    score = 0
    rule = model.rule
    observation, _ = env.reset(seed=seed)

    for _ in range(500):
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

