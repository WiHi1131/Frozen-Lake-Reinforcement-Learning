import gymnasium as gym  # Importing the Gym library for creating and managing environments
import numpy as np  # Importing NumPy for numerical operations
import random  # Importing random for stochastic elements in Q-Learning

class QLearningAgent:
    def __init__(self, env, learning_rate=0.8, discount_factor=0.95, exploration_rate=1.0, max_exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.005):
        self.env = env  # The environment in which the agent operates
        self.learning_rate = learning_rate  # Learning rate for Q-learning updates
        self.discount_factor = discount_factor  # Discount factor for future rewards
        self.exploration_rate = exploration_rate  # Initial exploration rate
        self.max_exploration_rate = max_exploration_rate  # Maximum exploration rate
        self.min_exploration_rate = min_exploration_rate  # Minimum exploration rate
        self.exploration_decay_rate = exploration_decay_rate  # Rate at which exploration rate decays
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))  # Initialize Q-table with zeros

    def train(self, total_episodes, max_steps):
        rewards = []  # To store rewards obtained in each episode

        # Training loop over episodes
        for episode in range(total_episodes):
            state, _ = self.env.reset()  # Reset the environment
            total_rewards = 0  # Initialize total rewards for the episode

            # Loop for each step in an episode
            for step in range(max_steps):
                exp_exp_tradeoff = random.uniform(0, 1)  # Exploration-exploitation decision
                # Choose action based on exploration rate or Q-table
                if exp_exp_tradeoff > self.exploration_rate:
                    action = np.argmax(self.q_table[state, :])  # Exploitation (choosing best action)
                else:
                    action = self.env.action_space.sample()  # Exploration (choosing random action)

                # Perform action and get new state and reward
                new_state, reward, terminated, truncated, _ = self.env.step(action)

                # Update Q-table using the Q-learning algorithm
                self.q_table[state, action] = self.q_table[state, action] + \
                    self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[new_state, :]) - self.q_table[state, action])

                total_rewards += reward  # Update total rewards
                state = new_state  # Update state

                # Break if the episode has ended
                if terminated or truncated:
                    break

            # Adjust the exploration rate
            self.exploration_rate = self.min_exploration_rate + \
                (self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * episode)
            rewards.append(total_rewards)  # Store rewards for this episode

        print("Score over time: " + str(sum(rewards) / total_episodes))  # Print average reward

    def test(self, total_episodes):
        # Testing loop over episodes
        for episode in range(total_episodes):
            state, _ = self.env.reset()  # Reset the environment
            done = False
            total_reward = 0

            # Loop until the episode ends
            while not done:
                action = self.get_action(state)  # Choose action based on Q-table
                state, reward, terminated, truncated, _ = self.env.step(action)  # Perform action
                total_reward += reward  # Update total reward
                done = terminated or truncated  # Check if episode ended
                self.env.render()  # Render the environment

            print(f"Episode {episode+1}: Total Reward: {total_reward}")  # Print total reward for the episode

    def get_action(self, state):
        return np.argmax(self.q_table[state, :])  # Choose the best action based on Q-table

    def play(self, total_episodes, max_steps):
        # Play loop over episodes
        for episode in range(total_episodes):
            state, _ = self.env.reset()  # Reset the environment
            print("****************************************************")
            print("EPISODE ", episode)

            # Loop for each step in an episode
            for step in range(max_steps):
                action = np.argmax(self.q_table[state, :])  # Choose the best action based on Q-table
                new_state, reward, terminated, truncated, _ = self.env.step(action)  # Perform action

                done = terminated or truncated  # Check if episode ended
                if done:
                    self.env.render()  # Render the environment
                    print("Number of steps", step)  # Print number of steps taken
                    break
                state = new_state  # Update state

        self.env.close()  # Close the environment

# Main execution block
if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', render_mode=None)  # Create environment without rendering for training
    agent = QLearningAgent(env)  # Initialize QLearningAgent
    agent.train(15000, 99)  # Train the agent
    env.close()  # Close the environment

    env = gym.make('FrozenLake-v1', render_mode="human")  # Create environment with rendering for playing
    agent.env = env  # Update the agent's environment
    agent.play(5, 99)  # Play the game using the trained Q-table
    env.close()  # Close the environment




