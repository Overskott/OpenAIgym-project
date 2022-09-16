import gym
import policies

env = gym.make("CartPole-v1", render_mode='human')
observation, _ = env.reset() # 546 gives full score with rule 81

rule = 81 # 97
score = 0

for _ in range(500):
    action = policies.simple_ca(observation, rule)  # User-defined policy function
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