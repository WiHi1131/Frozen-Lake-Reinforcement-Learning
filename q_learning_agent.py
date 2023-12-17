import gymnasium as gym
import numpy as np
import random

class QLearningAgent:
    def __init__(self, env, learning_rate=0.8, discount_factor=0.95, exploration_rate=1.0, max_exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.005):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def train(self, total_episodes, max_steps):
        rewards = []
        for episode in range(total_episodes):
            state, _ = self.env.reset()
            total_rewards = 0

            for step in range(max_steps):
                exp_exp_tradeoff = random.uniform(0, 1)
                if exp_exp_tradeoff > self.exploration_rate:
                    action = np.argmax(self.q_table[state, :])
                else:
                    action = self.env.action_space.sample()

                new_state, reward, terminated, truncated, _ = self.env.step(action)

                # Check if the episode should end
                done = terminated or truncated

                # Update Q-Table
                self.q_table[state, action] = self.q_table[state, action] + \
                    self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[new_state, :]) - self.q_table[state, action])

                total_rewards += reward
                state = new_state

                if done:
                    break

            self.exploration_rate = self.min_exploration_rate + \
                (self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * episode)
            rewards.append(total_rewards)

        print("Score over time: " + str(sum(rewards) / total_episodes))

    def test(self, total_episodes):
        for episode in range(total_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.get_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                self.env.render()

            print(f"Episode {episode+1}: Total Reward: {total_reward}")

    def get_action(self, state):
        return np.argmax(self.q_table[state, :])
    
    def play(self, total_episodes, max_steps):
        for episode in range(total_episodes):
            state, _ = self.env.reset()
            print("****************************************************")
            print("EPISODE ", episode)

            for step in range(max_steps):
                action = np.argmax(self.q_table[state, :])
                new_state, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated
                if done:
                    self.env.render()
                    print("Number of steps", step)
                    break
                state = new_state

        self.env.close()

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', render_mode=None)
    agent = QLearningAgent(env)
    agent.train(15000, 99)  # Updated number of training episodes and max_steps
    env.close()

    env = gym.make('FrozenLake-v1', render_mode="human")
    agent.env = env
    agent.play(5, 99)  # Using the Q-table as a 'cheatsheet' to play
    env.close()



