import matplotlib.style
import numpy as np
from myenv import MyEnv
from collections import defaultdict
import plotting
import random

matplotlib.style.use('ggplot')


def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
    """
    Creates an epsilon-greedy policy based
    on a given Q-function and epsilon.

    Returns a function that takes the state
    as an input and returns the probabilities
    for each action in the form of a numpy array
    of length of the action space(set of possible actions).
    """

    def policyFunction(state):
        Action_probabilities = np.ones(num_actions,
                                       dtype=float) * epsilon / num_actions

        best_action = np.argmax(Q[str(state)])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities

    return policyFunction


def qLearning(env, num_episodes, discount_factor=1.0,
              alpha=0.6, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""

    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.no_actions))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createEpsilonGreedyPolicy(Q, epsilon, env.no_actions)

    # For every episode
    for ith_episode in range(num_episodes):

        # Reset the environment and pick the first action
        state = env.reset()

        for i in range(env.number_of_vehicles):

            # get probabilities of all actions from current state
            action_probabilities = policy(state)

            # choose action according to
            # the probability distribution
            action = np.random.choice(np.arange(
                len(action_probabilities)),
                p=action_probabilities)

            # take action and get reward, transit to next state
            reward, next_state, done = env.step(action)
            # print(state)
            # print(action)
            # print(next_state)
            # print("___________________________________________")
            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = i

            # TD Update
            best_next_action = np.argmax(Q[str(next_state)])
            td_target = reward + discount_factor * Q[str(next_state)][best_next_action]
            td_delta = td_target - Q[str(state)][action]
            Q[str(state)][action] += alpha * td_delta

            # done is True if episode terminated
            if done:
                break

            state = next_state

    return Q, stats


densities = [0.002]  # 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004

plot_rewards = []
plot_powers = []
plot_services = []

plot_rewards_greedy = []
plot_powers_greedy = []
plot_services_greedy = []

plot_rewards_random = []
plot_powers_random = []
plot_services_random = []

plot_rewards_most = []
plot_powers_most = []
plot_services_most = []

plot_rewards_static = []
plot_powers_static = []
plot_services_static = []

for density in densities:
    iterations = 1
    # All best rewards for all iterations
    all_rewards = []
    all_powers = []
    all_services = []

    all_rewards_greedy = []
    all_powers_greedy = []
    all_services_greedy = []

    all_rewards_random = []
    all_powers_random = []
    all_services_random = []

    all_rewards_most = []
    all_powers_most = []
    all_services_most = []

    all_rewards_static = []
    all_powers_static = []
    all_services_static = []

    for iteration in range(iterations):

        myenv = MyEnv(density=density)
        myenv.RSU_cache_size = 100
        # 1) Greedy Algorithm
        myenv.reset()
        for i in range(myenv.number_of_vehicles):
            myenv.step(1);

        # Store the best reward for one iteration
        all_rewards_greedy.append(myenv.total_reward)
        all_powers_greedy.append(myenv.total_energy)
        all_services_greedy.append(myenv.total_download / myenv.total_request_amount)

        # 2) Random Algorithm
        myenv.reset()
        actions = [random.randrange(0, 2, 1) for i in range(myenv.number_of_vehicles)]
        for i in range(myenv.number_of_vehicles):
            myenv.step(actions[i]);

        # Store the best reward
        all_rewards_random.append(myenv.total_reward)
        all_powers_random.append(myenv.total_energy)
        all_services_random.append(myenv.total_download / myenv.total_request_amount)

        # 3) Most Popular Algorithm
        myenv.reset()

        valid_to_cache = []
        taken = 0
        for c in range(myenv.number_of_contents):
            if myenv.contents_sizes[c] < myenv.RSU_cache_size - taken:
                valid_to_cache.append(c);
                taken += myenv.contents_sizes[c]

        for i in range(myenv.number_of_vehicles):
            if (myenv.available[i] in valid_to_cache):
                myenv.step(1)
            else:
                myenv.step(0)

        # Store the best reward
        all_rewards_most.append(myenv.total_reward)
        all_powers_most.append(myenv.total_energy)
        all_services_most.append(myenv.total_download / myenv.total_request_amount)

        # 4) Most Popular Algorithm
        myenv.reset()

        taken = 0

        for i in range(myenv.number_of_vehicles):
            if myenv.contents_sizes[myenv.available[i]] < myenv.RSU_cache_size - taken:
                myenv.step(1)
                taken += myenv.contents_sizes[myenv.available[i]]
            else:
                myenv.step(0)

        # Store the best reward
        all_rewards_static.append(myenv.total_reward)
        all_powers_static.append(myenv.total_energy)
        all_services_static.append(myenv.total_download / myenv.total_request_amount)

        # 5) RL Algorithm
        Q, stats = qLearning(myenv, 5000)

        all_rewards.append(myenv.total_reward)
        all_powers.append(myenv.total_energy)
        all_services.append(myenv.total_download / myenv.total_request_amount)
        # Store the best reward for one iteration

    print("Number of Cars=" + str(myenv.number_of_vehicles) + " | Density=" + str(myenv.density))
    print("**Reward RL :" + str(sum(all_rewards) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers) / iterations)
          + " | " + "Service Rate:" + str(sum(all_services) / iterations))

    print("**Reward Greedy:" + str(sum(all_rewards_greedy) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers_greedy) / iterations)
          + " | " + "Service Rate:" + str(sum(all_services_greedy) / iterations))

    print("**Reward Random:" + str(sum(all_rewards_random) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers_random) / iterations)
          + " | " + "Service Rate:" + str(sum(all_services_random) / iterations))

    print("**Reward MOST:" + str(sum(all_rewards_most) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers_most) / iterations)
          + " | " + "Service Rate:" + str(sum(all_services_most) / iterations))

    print("**Reward Static:" + str(sum(all_rewards_static) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers_static) / iterations)
          + " | " + "Service Rate:" + str(sum(all_services_static) / iterations))
    print("____________________________________________________" + "\n")

    f = open("result.txt", "a")
    f.write("Number of Cars=" + str(myenv.number_of_vehicles) + " | Density=" + str(myenv.density) + "\n")
    f.write("**Reward RL :" + str(sum(all_rewards) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers) / iterations)
            + " | " + "Service Rate:" + str(sum(all_services) / iterations) + "\n")

    f.write("**Reward Greedy:" + str(sum(all_rewards_greedy) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers_greedy) / iterations)
            + " | " + "Service Rate:" + str(sum(all_services_greedy) / iterations) + "\n")

    f.write("**Reward Random:" + str(sum(all_rewards_random) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers_random) / iterations)
            + " | " + "Service Rate:" + str(sum(all_services_random) / iterations) + "\n")

    f.write("**Reward MOST:" + str(sum(all_rewards_most) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers_most) / iterations)
            + " | " + "Service Rate:" + str(sum(all_services_most) / iterations) + "\n")

    f.write("**Reward Static:" + str(sum(all_rewards_static) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers_static) / iterations)
            + " | " + "Service Rate:" + str(sum(all_services_static) / iterations) + "\n")

    f.write("____________________________________________________" + "\n")
    f.close()

    plot_rewards.append(sum(all_rewards) / iterations)
    plot_powers.append(sum(all_powers) / iterations)
    plot_services.append(sum(all_services) / iterations)

    plot_rewards_greedy.append(sum(all_rewards_greedy) / iterations)
    plot_powers_greedy.append(sum(all_powers_greedy) / iterations)
    plot_services_greedy.append(sum(all_services_greedy) / iterations)

    plot_rewards_random.append(sum(all_rewards_random) / iterations)
    plot_powers_random.append(sum(all_powers_random) / iterations)
    plot_services_random.append(sum(all_services_random) / iterations)

    plot_rewards_most.append(sum(all_rewards_most) / iterations)
    plot_powers_most.append(sum(all_powers_most) / iterations)
    plot_services_most.append(sum(all_services_most) / iterations)

    plot_rewards_static.append(sum(all_rewards_static) / iterations)
    plot_powers_static.append(sum(all_powers_static) / iterations)
    plot_services_static.append(sum(all_services_static) / iterations)

    plotting.plot_episode_stats(stats, smoothing_window=50)

f = open("result.txt", "a")
f.write(str(plot_rewards) + "\n")
f.write(str(plot_powers) + "\n")
f.write(str(plot_services) + "\n")

f.write(str(plot_rewards_greedy) + "\n")
f.write(str(plot_powers_greedy) + "\n")
f.write(str(plot_services_greedy) + "\n")

f.write(str(plot_rewards_random) + "\n")
f.write(str(plot_powers_random) + "\n")
f.write(str(plot_services_random) + "\n")

f.write(str(plot_rewards_most) + "\n")
f.write(str(plot_powers_most) + "\n")
f.write(str(plot_services_most) + "\n")

f.write(str(plot_rewards_static) + "\n")
f.write(str(plot_powers_static) + "\n")
f.write(str(plot_services_static) + "\n")
f.write("________________________END EXPERMINT____________________________" + "\n")
