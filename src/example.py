import gym
import policies
import utils

rule = utils.int_to_binary(90)
print(rule)
score = 0
seed = 5354

env = gym.make("CartPole-v1", render_mode='human')
observation, _ = env.reset(seed=seed)

for _ in range(500):
    # Best run_time_evolution was run_time_evolution #112 with the score 500.0 using _rule 112 with seed 1921
    action = policies.simple_ca(observation, rule)  # 546 gives full score with _rule 81
    #action = policies.spread_out(observation, _rule)  # Run 106, score 144.0, _rule 106 seed 1707 _size 100
                                                     # Best run_time_evolution was run_time_evolution #28 with the score 261.0 using _rule 28 with seed 2719

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