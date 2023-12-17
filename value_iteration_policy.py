import gymnasium as gym
import numpy as np

def run_value_iteration(env, discount_factor=0.9, theta=1e-4, print_iterations=[100, 1000]):
    value_table = np.zeros(env.observation_space.n)
    policy_table = np.zeros(env.observation_space.n, dtype=int)
    iteration = 0

    while True:
        delta = 0
        iteration += 1
        for state in range(env.observation_space.n):
            v = value_table[state]
            value_table[state] = max(sum(prob * (reward + discount_factor * value_table[next_state])
                                        for prob, next_state, reward, _ in env.P[state][action])
                                    for action in range(env.action_space.n))
            delta = max(delta, abs(v - value_table[state]))

        if iteration in print_iterations:
            print(f"Value Table after {iteration} iterations:")
            print(value_table)

        if delta < theta:
            break

    for state in range(env.observation_space.n):
        policy_table[state] = np.argmax([sum(prob * (reward + discount_factor * value_table[next_state])
                                             for prob, next_state, reward, _ in env.P[state][action])
                                         for action in range(env.action_space.n)])

    return policy_table

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', render_mode="human")
    policy = run_value_iteration(env)
    env.close()




