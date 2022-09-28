import gym
import policies


rule = 71
score = 0
seed = 5354

env = gym.make("CartPole-v1", render_mode='human')
observation, _ = env.reset(seed=seed)

for _ in range(500):
    # Best run was run #112 with the score 500.0 using rule 112 with seed 1921
    action = policies.spread_out(observation, rule)  # 546 gives full score with rule 81
    #action = policies.spread_out(observation, rule)  # Run 106, score 144.0, rule 106 seed 1707 size 100
                                                     # Best run was run #28 with the score 261.0 using rule 28 with seed 2719

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