import gymnasium as gym
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, env, learning_rate=0.001, discount_factor=0.95, exploration_rate=1.0, max_exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.005):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

        # Simple Neural Network for Q-Learning
        self.model = Sequential([
            Dense(24, input_shape=(env.observation_space.n,), activation='relu'),
            Dense(24, activation='relu'),
            Dense(env.action_space.n, activation='linear')
        ])
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

    def train(self, total_episodes, max_steps):
        for episode in range(total_episodes):
            state, _ = self.env.reset()  # Extract state from the tuple
            state_one_hot = np.identity(self.env.observation_space.n)[state:state+1]
            total_rewards = 0

            for step in range(max_steps):
                if random.uniform(0, 1) > self.exploration_rate:
                    action = np.argmax(self.model.predict(state_one_hot)[0])
                else:
                    action = self.env.action_space.sample()

                new_state, reward, terminated, truncated, _ = self.env.step(action)
                new_state_one_hot = np.identity(self.env.observation_space.n)[new_state:new_state+1]

                target = (reward + self.discount_factor * 
                        np.max(self.model.predict(new_state_one_hot)[0]))
                target_f = self.model.predict(state_one_hot)
                target_f[0][action] = target

                self.model.fit(state_one_hot, target_f, epochs=1, verbose=0)
                total_rewards += reward
                state_one_hot = new_state_one_hot

                if terminated or truncated:
                    break

            self.exploration_rate = self.min_exploration_rate + \
                (self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * episode)

            print(f"Episode {episode+1}: Total Reward: {total_rewards}")

    def get_action(self, state):
        state = np.identity(self.env.observation_space.n)[state:state+1]
        return np.argmax(self.model.predict(state)[0])

    # Include your play function here for testing
    
    def play(self, total_episodes):
        for episode in range(total_episodes):
            state, _ = self.env.reset()
            state_one_hot = np.identity(self.env.observation_space.n)[state:state+1]
            done = False
            step = 0

            print("****************************************************")
            print(f"EPISODE {episode + 1}")

            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state_one_hot)[0])
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                new_state_one_hot = np.identity(self.env.observation_space.n)[new_state:new_state+1]

                state_one_hot = new_state_one_hot
                done = terminated or truncated
                step += 1

            print(f"Episode {episode + 1} finished after {step} steps")

        self.env.close()

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', render_mode=None)
    agent = DQNAgent(env)
    agent.train(5000, 99)  # Training episodes and max_steps
    env.close()

    env = gym.make('FrozenLake-v1', render_mode="human")
    agent.env = env
    agent.play(5)  # Play 5 episodes in human mode
    env.close()
    # Add code to test the agent in 'human' mode
