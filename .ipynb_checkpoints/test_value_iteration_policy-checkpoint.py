import gymnasium as gym  # Importing the Gym library for creating and managing environments
from value_iteration_policy import run_value_iteration  # Importing the value iteration function

# Function to test the policy in the environment
def test_policy(env, policy, total_episodes=100):
    total_rewards = 0  # Initialize total rewards

    # Run the policy for a specified number of episodes
    for episode in range(total_episodes):
        observation, info = env.reset()  # Reset the environment at the start of each episode
        episode_reward = 0  # Initialize reward for this episode

        # Run the episode for a maximum of 99 steps
        for _ in range(99):
            action = policy[observation]  # Select an action based on the policy
            # Perform the action in the environment and get the next state and reward
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward  # Accumulate reward

            # Break the loop if the episode is terminated or truncated
            if terminated or truncated:
                break
        total_rewards += episode_reward  # Add episode reward to total rewards

    average_reward = total_rewards / total_episodes  # Calculate the average reward
    return average_reward  # Return the average reward

# Function to render and demonstrate the policy in the environment
def render_policy(env, policy, total_episodes=5):
    for episode in range(total_episodes):  # Loop for a specified number of episodes
        observation, info = env.reset()  # Reset the environment at the start of each episode

        # Run the episode for a maximum of 99 steps
        for step in range(99):
            env.render()  # Render the current state of the environment
            action = policy[observation]  # Select an action based on the policy
            # Perform the action in the environment and get the next state and reward
            observation, reward, terminated, truncated, info = env.step(action)

            # Break the loop if the episode is terminated or truncated
            if terminated or truncated:
                print("****************************************************")
                print(f"EPISODE {episode + 1}")
                print("Number of steps:", step)
                break

# Main execution block
if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', render_mode=None)  # Create environment without rendering for training
    policy = run_value_iteration(env)  # Run value iteration to get the policy

    # Test the policy and print the average reward
    average_reward = test_policy(env, policy, 100)
    print("Average Score over time: " + str(average_reward))

    # Create environment with rendering for demonstration
    env = gym.make('FrozenLake-v1', render_mode="human")
    render_policy(env, policy, 5)  # Render and demonstrate the policy
    env.close()  # Close the environment
