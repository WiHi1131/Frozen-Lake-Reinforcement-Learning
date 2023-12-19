import gymnasium as gym
from value_iteration_policy import run_value_iteration

def test_policy(env, policy, total_episodes=100):
    total_rewards = 0
    for episode in range(total_episodes):
        observation, info = env.reset()
        episode_reward = 0
        for _ in range(199):  # Max steps per episode
            action = policy[observation]
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break
        total_rewards += episode_reward

    average_reward = total_rewards / total_episodes
    return average_reward

def render_policy(env, policy, total_episodes=5):
    for episode in range(total_episodes):
        observation, info = env.reset()
        for step in range(199):  # Max steps per episode
            env.render()
            action = policy[observation]
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print("****************************************************")
                print(f"EPISODE {episode + 1}")
                print("Number of steps:", step)
                break

if __name__ == "__main__":
    env = gym.make('FrozenLake8x8-v1', render_mode=None)  # Train without rendering
    policy = run_value_iteration(env)

    # Test the policy and print average reward
    average_reward = test_policy(env, policy, 100)
    print("Average Score over time: " + str(average_reward))

    # Render in human mode for demonstration and print steps per episode
    env = gym.make('FrozenLake8x8-v1', render_mode="human")
    render_policy(env, policy, 5)
    env.close()