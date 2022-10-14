import gym
import policies
import utils
from genotypes import CellularAutomaton1D

rule = utils.int_to_binary(3002457800, 2**5)
print(rule)
score = 0
seed = 5354
size = 10

env = gym.make("CartPole-v1", render_mode='human')
observation, _ = env.reset()

for _ in range(500):
    model = CellularAutomaton1D(rule=rule, size=size, hood_size=5)

    action = policies.simple_ca(observation, model)  # User-defined policy function
    print(action)
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

print(f"Score {score}")