import gymnasium as gym
from q_learning_agent import QLearningAgent

def test_policy(env, agent, num_episodes=1000):
    total_rewards = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            total_rewards += reward

        print(f"Episode {episode+1}: Total Reward: {total_rewards}")
        total_rewards = 0

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', render_mode="human", is_slippery=True)
    agent = QLearningAgent(env)
    agent.train(10000)  # Training with 10000 episodes
    test_policy(env, agent)
    env.close()
