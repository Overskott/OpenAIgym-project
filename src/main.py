#
#  https://www.gymlibrary.dev/environments/classic_control/cart_pole/
#
#
import gym
import time
import policies

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

print(['Position', 'Velocity', 'Angle', 'Ang. vel.'])
score = 0
for _ in range(50):
    action = policies.naive_control(observation)  # User-defined policy function
    observation, reward, terminated, truncated, info = env.step(action)
    score += reward
    print(f"reward: {reward}, terminated: {terminated}, turncated: {truncated}, info: {info}")
    print(observation)
    if terminated:
        observation, info = env.reset()
        print(f"Run ended  with score {score}")
        break
    if terminated:
        observation, info = env.reset()
    time.sleep(0.05)

env.close()
