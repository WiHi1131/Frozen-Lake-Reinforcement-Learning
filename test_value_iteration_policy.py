import gymnasium as gym
from value_iteration_policy import run_value_iteration

def test_policy(env, policy):
    observation, info = env.reset()
    total_reward = 0
    for _ in range(1000):
        action = policy[observation]
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            observation, info = env.reset()
            print(f"Total Reward: {total_reward}")
            total_reward = 0

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', render_mode="human")
    policy = run_value_iteration(env)
    test_policy(env, policy)
    env.close()
