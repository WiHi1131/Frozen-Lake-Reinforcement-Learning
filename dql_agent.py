import gymnasium as gym  # Importing the Gym library for creating and managing environments
import numpy as np  # Importing NumPy for numerical operations
import random  # Importing random for stochastic elements in Deep Q-Learning
import tensorflow as tf  # Importing TensorFlow for building neural network
from tensorflow.keras.models import Sequential  # Sequential model for creating neural network
from tensorflow.keras.layers import Dense  # Dense layer for neural network
from tensorflow.keras.optimizers import Adam  # Adam optimizer for training neural network

class DQNAgent:
    def __init__(self, env, learning_rate=0.001, discount_factor=0.95, exploration_rate=1.0, max_exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.005):
        self.env = env  # The environment in which the agent operates
        self.learning_rate = learning_rate  # Learning rate for neural network
        self.discount_factor = discount_factor  # Discount factor for future rewards
        self.exploration_rate = exploration_rate  # Initial exploration rate
        self.max_exploration_rate = max_exploration_rate  # Maximum exploration rate
        self.min_exploration_rate = min_exploration_rate  # Minimum exploration rate
        self.exploration_decay_rate = exploration_decay_rate  # Rate at which exploration rate decays

        # Neural Network model for Deep Q-Learning
        self.model = Sequential([
            Dense(24, input_shape=(env.observation_space.n,), activation='relu'),  # First hidden layer
            Dense(24, activation='relu'),  # Second hidden layer
            Dense(env.action_space.n, activation='linear')  # Output layer
        ])
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))  # Compile the model

    def train(self, total_episodes, max_steps):
        # Training loop over episodes
        for episode in range(total_episodes):
            state, _ = self.env.reset()  # Reset the environment
            state_one_hot = np.identity(self.env.observation_space.n)[state:state+1]  # One-hot encode state
            total_rewards = 0  # Initialize total rewards for the episode

            # Loop for each step in an episode
            for step in range(max_steps):
                # Exploration-exploitation decision
                if random.uniform(0, 1) > self.exploration_rate:
                    action = np.argmax(self.model.predict(state_one_hot)[0])  # Exploitation
                else:
                    action = self.env.action_space.sample()  # Exploration

                # Perform action and get new state and reward
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                new_state_one_hot = np.identity(self.env.observation_space.n)[new_state:new_state+1]  # One-hot encode new state

                # Update target for Q-value
                target = (reward + self.discount_factor * 
                        np.max(self.model.predict(new_state_one_hot)[0]))
                target_f = self.model.predict(state_one_hot)
                target_f[0][action] = target

                # Fit the model
                self.model.fit(state_one_hot, target_f, epochs=1, verbose=0)
                total_rewards += reward  # Update total rewards
                state_one_hot = new_state_one_hot  # Update state

                # Break if the episode has ended
                if terminated or truncated:
                    break

            # Adjust the exploration rate
            self.exploration_rate = self.min_exploration_rate + \
                (self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * episode)

            print(f"Episode {episode+1}: Total Reward: {total_rewards}")  # Print reward for the episode

    def get_action(self, state):
        state_one_hot = np.identity(self.env.observation_space.n)[state:state+1]  # One-hot encode state
        return np.argmax(self.model.predict(state_one_hot)[0])  # Choose action based on Q-values

    def play(self, total_episodes):
        # Play loop over episodes
        for episode in range(total_episodes):
            state, _ = self.env.reset()  # Reset the environment
            state_one_hot = np.identity(self.env.observation_space.n)[state:state+1]  # One-hot encode state
            done = False
            step = 0

            print("****************************************************")
            print(f"EPISODE {episode + 1}")

            while not done:
                self.env.render()  # Render the environment
                action = np.argmax(self.model.predict(state_one_hot)[0])  # Choose action based on Q-values
                new_state, reward, terminated, truncated, _ = self.env.step(action)  # Perform action
                new_state_one_hot = np.identity(self.env.observation_space.n)[new_state:new_state+1]  # One-hot encode new state

                state_one_hot = new_state_one_hot  # Update state
                done = terminated or truncated  # Check if episode ended
                step += 1

            print(f"Episode {episode + 1} finished after {step} steps")  # Print steps taken

        self.env.close()  # Close the environment

# Main execution block
if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', render_mode=None)  # Create environment without rendering for training
    agent = DQNAgent(env)  # Initialize DQNAgent
    agent.train(5000, 99)  # Train the agent
    env.close()  # Close the environment

    env = gym.make('FrozenLake-v1', render_mode="human")  # Create environment with rendering for playing
    agent.env = env  # Update the agent's environment
    agent.play(5)  # Play 5 episodes in human mode
    env.close()  # Close the environment
