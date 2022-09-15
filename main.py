import gym
import time
import random

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

policy = lambda obs: random.randint(0, 1)


for _ in range(100):
   action = policy(observation)  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   #if terminated or truncated:
   #   observation, info = env.reset()
   time.sleep(0.05)
env.close()