import gymnasium as gym  # Importing the Gym library for creating and managing environments
import numpy as np  # Importing NumPy for numerical operations

# Define the function for running value iteration
def run_value_iteration(env, discount_factor=0.9, theta=1e-4, print_iterations=[100, 1000]):
    value_table = np.zeros(env.observation_space.n)  # Initialize the value table with zeros
    policy_table = np.zeros(env.observation_space.n, dtype=int)  # Initialize the policy table with zeros
    iteration = 0  # Counter for iterations

    while True:  # Start of the value iteration loop
        delta = 0  # Initialize the delta, which tracks the change in value
        iteration += 1  # Increment the iteration count

        # Loop over all states in the environment
        for state in range(env.observation_space.n):
            v = value_table[state]  # Store the current value of the state
            # Update the value of the state based on the Bellman equation
            value_table[state] = max(sum(prob * (reward + discount_factor * value_table[next_state])
                                        for prob, next_state, reward, _ in env.P[state][action])
                                    for action in range(env.action_space.n))
            # Update delta with the maximum change observed in the value table
            delta = max(delta, abs(v - value_table[state]))

        # Print the value table at specified iterations
        if iteration in print_iterations:
            print(f"Value Table after {iteration} iterations:")
            print(value_table)

        # Check for convergence, break if the change is below the threshold
        if delta < theta:
            break

    # Policy extraction loop
    for state in range(env.observation_space.n):
        # For each state, find the best action by looking at the future rewards
        policy_table[state] = np.argmax([sum(prob * (reward + discount_factor * value_table[next_state])
                                             for prob, next_state, reward, _ in env.P[state][action])
                                         for action in range(env.action_space.n)])

    return policy_table  # Return the final policy table

# Main execution block
if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', render_mode="human")  # Create the FrozenLake environment
    policy = run_value_iteration(env)  # Run value iteration on the environment
    env.close()  # Close the environment after running value iteration




