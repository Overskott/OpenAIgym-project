#
#  https://www.gymlibrary.dev/environments/classic_control/cart_pole/
#
#
import gym
import policies

env = gym.make("CartPole-v1")

run = 0
results = [0, 0, 0]

for run in range(258):
    score = 0
    rule = run
    observation, _ = env.reset(seed=546)

    for _ in range(500):
        action = policies.voter_control(observation)  # User-defined policy function
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

    print(f"Run {run}, score {score}, rule {rule}")

    if score > results[1]:
        results[0] = run
        results[1] = score
        results[2] = rule

print(f"Best run was run #{results[0]} with the score {results[1]} using rule {results[2]}")