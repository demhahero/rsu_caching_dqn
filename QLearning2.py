import numpy as np
from myenv import MyEnv
from collections import defaultdict
import plotting
import random
import copy

random.seed(1)
np.random.seed(1)

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


def qLearning(env, num_episodes, start, discount_factor=0.9,
              alpha=0.9, epsilon=0.1):
    best_reward =  copy.copy(env);
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""

    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.no_actions))
        #print("nono")

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

            if(i == start):
                env.reset(dont=True)
                env.i = start

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

        if(best_reward.total_reward < env.total_reward):
            best_reward =  copy.copy(env);

        if (ith_episode % 500 == 0):
            print(str(ith_episode));
    return Q, stats, best_reward


densities = [0.006]  # 0.001, 0.002, 0.003, 0.004, 0.005, 0.006
#caches = [1000]
#ws = [0, 0.2, 0.4, 0.6, 0.8, 1] #0, 0.2, 0.4, 0.6, 0.8, 1
numbers = [20]

time = 1000 #21600
no_iterations = 2000

plot_rewards = []
plot_powers = []
plot_services = []
plot_upload = []

plot_rewards_greedy = []
plot_powers_greedy = []
plot_services_greedy = []
plot_upload_greedy = []

plot_rewards_random = []
plot_powers_random = []
plot_services_random = []
plot_upload_random = []

plot_rewards_most = []
plot_powers_most = []
plot_services_most = []
plot_upload_most = []

plot_rewards_min = []
plot_powers_min = []
plot_services_min = []
plot_upload_min = []

plot_rewards_max = []
plot_powers_max = []
plot_services_max = []
plot_upload_max = []

cache = -1



for density in densities:



    iterations = 1
    # All best rewards for all iterations
    all_rewards = []
    all_powers = []
    all_services = []
    all_upload = []

    all_rewards_greedy = []
    all_powers_greedy = []
    all_services_greedy = []
    all_upload_greedy = []

    all_rewards_random = []
    all_powers_random = []
    all_services_random = []
    all_upload_random = []

    all_rewards_most = []
    all_powers_most = []
    all_services_most = []
    all_upload_most = []

    all_rewards_min = []
    all_powers_min = []
    all_services_min = []
    all_upload_min = []

    all_rewards_max = []
    all_powers_max = []
    all_services_max = []
    all_upload_max = []

    # Q = None;
    # myenv = MyEnv(density=density, T=100000)
    # print("learning:"+str(myenv.number_of_vehicles))
    #
    # Q, stats = qLearning(myenv, 1)

    cache += 1
    for iteration in range(iterations):
        myenv = MyEnv(density=density, T=time, number_of_contents=numbers[cache])
        #myenv.RSU_cache_size = caches[cache]
        #myenv.hit_energy_ratio = ws[cache]


        print("Testing:" + str(myenv.number_of_vehicles))
        # 1) Greedy Algorithm
        myenv.reset()
        start = 0

        myenv.i = start
        for i in range(start, myenv.number_of_vehicles):
            if(i not in myenv.available_contents_to_cache and myenv.available[i] not in myenv.RSU_cache):
                myenv.step(1);
            else:
                myenv.step(0);

        # Store the best reward for one iteration
        all_rewards_greedy.append(int(myenv.total_reward))
        all_powers_greedy.append(int(myenv.total_energy))
        all_services_greedy.append(myenv.total_download/myenv.total_request_amount)
        all_upload_greedy.append(int(myenv.RSU_cache_upload))

        # 2) Random Algorithm
        myenv.reset()
        myenv.i = start
        #actions = [random.randrange(0, 2, 1) for i in range(myenv.number_of_vehicles)]

        actions = np.random.choice([0, 1], size=(myenv.number_of_vehicles,), p=[1./3, 2./3])

        for i in range(start, myenv.number_of_vehicles):
            if(i not in myenv.available_contents_to_cache and myenv.available[i] not in myenv.RSU_cache):
                myenv.step(actions[i]);
            else:
                myenv.step(0);

        # Store the best reward
        all_rewards_random.append(int(myenv.total_reward))
        all_powers_random.append(int(myenv.total_energy))
        all_services_random.append(myenv.total_download/myenv.total_request_amount)
        all_upload_random.append(int(myenv.RSU_cache_upload))

        # 3) Most Popular Algorithm
        myenv.reset()
        myenv.i = start
        valid_to_cache = []
        taken = 0
        for c in range(myenv.number_of_contents):
            if myenv.contents_sizes[c] < myenv.RSU_cache_size - taken:
                valid_to_cache.append(c);
                taken += myenv.contents_sizes[c]

        for i in range(start, myenv.number_of_vehicles):
            if (myenv.available[i] in valid_to_cache and (i not in myenv.available_contents_to_cache and myenv.available[i] not in myenv.RSU_cache)):
                myenv.step(1)
            else:
                myenv.step(0)

        # Store the best reward
        all_rewards_most.append(int(myenv.total_reward))
        all_powers_most.append(int(myenv.total_energy))
        all_services_most.append(myenv.total_download/myenv.total_request_amount)
        all_upload_most.append(int(myenv.RSU_cache_upload))

        # 4) Min
        myenv.reset()
        myenv.i = start
        taken = 0

        for i in range(start, myenv.number_of_vehicles):
            if (i not in myenv.available_contents_to_cache and myenv.available[i] not in myenv.RSU_cache):
                if myenv.contents_sizes[myenv.available[i]] < myenv.RSU_cache_size - taken:
                    myenv.step(1)
                    taken += myenv.contents_sizes[myenv.available[i]]
                else:
                    cached_sizes = []
                    for c in myenv.RSU_cache:
                        cached_sizes.append(myenv.contents_sizes[c])

                    for i in myenv.available_contents_to_cache:
                        cached_sizes.append(myenv.contents_sizes[myenv.available[i]])

                    if(myenv.contents_sizes[myenv.available[i]] < max(cached_sizes)):
                        myenv.step(1)
                    else:
                        myenv.step(0)
            else:
                myenv.step(0)

        # Store the best reward
        all_rewards_min.append(int(myenv.total_reward))
        all_powers_min.append(int(myenv.total_energy))
        all_services_min.append(myenv.total_download/myenv.total_request_amount)
        all_upload_min.append(int(myenv.RSU_cache_upload))

        # 5) Max
        myenv.reset()
        myenv.i = start
        taken = 0

        for i in range(start, myenv.number_of_vehicles):
            if (i not in myenv.available_contents_to_cache and myenv.available[i] not in myenv.RSU_cache):
                if myenv.contents_sizes[myenv.available[i]] < myenv.RSU_cache_size - taken:
                    myenv.step(1)
                    taken += myenv.contents_sizes[myenv.available[i]]
                else:
                    cached_sizes = []
                    for c in myenv.RSU_cache:
                        cached_sizes.append(myenv.contents_sizes[c])

                    for i in myenv.available_contents_to_cache:
                        cached_sizes.append(myenv.contents_sizes[myenv.available[i]])

                    if(myenv.contents_sizes[myenv.available[i]] > min(cached_sizes)):
                        myenv.step(1)
                    else:
                        myenv.step(0)
            else:
                myenv.step(0)

        # Store the best reward
        all_rewards_max.append(int(myenv.total_reward))
        all_powers_max.append(int(myenv.total_energy))
        all_services_max.append(myenv.total_download/myenv.total_request_amount)
        all_upload_max.append(int(myenv.RSU_cache_upload))

        # 6) RL Algorithm
        myenv.reset()
        Q, stats, myenv = qLearning(myenv, no_iterations, start=start)
        print("cache size="+str(myenv.RSU_cache_size))
        all_rewards.append(int(myenv.total_reward))
        all_powers.append(int(myenv.total_energy))
        all_services.append(myenv.total_download/myenv.total_request_amount)
        all_upload.append(int(myenv.RSU_cache_upload))
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

    print("**Reward Min:" + str(sum(all_rewards_min) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers_min) / iterations)
          + " | " + "Service Rate:" + str(sum(all_services_min) / iterations))

    print("**Reward Max:" + str(sum(all_rewards_max) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers_max) / iterations)
          + " | " + "Service Rate:" + str(sum(all_services_max) / iterations))
    print("____________________________________________________" + "\n")

    f = open("result.txt", "a")
    f.write("Number of Cars=" + str(myenv.number_of_vehicles) + " | Density=" + str(myenv.density) + "\n")
    f.write("**Reward RL :" + str(sum(all_rewards) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers) / iterations)
            + " | " + "Service Rate:" + str(sum(all_services) / iterations) + " | " + "Upload:" + str(sum(all_upload) / iterations) + "\n")

    f.write("**Reward Greedy:" + str(sum(all_rewards_greedy) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers_greedy) / iterations)
            + " | " + "Service Rate:" + str(sum(all_services_greedy) / iterations) +  " | " + "Upload:" + str(sum(all_upload_greedy) / iterations) + "\n")

    f.write("**Reward Random:" + str(sum(all_rewards_random) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers_random) / iterations)
            + " | " + "Service Rate:" + str(sum(all_services_random) / iterations) +  " | " + "Upload:" + str(sum(all_upload_random) / iterations) + "\n")

    f.write("**Reward MOST:" + str(sum(all_rewards_most) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers_most) / iterations)
            + " | " + "Service Rate:" + str(sum(all_services_most) / iterations) +  " | " + "Upload:" + str(sum(all_upload_most) / iterations) + "\n")

    f.write("**Reward Min:" + str(sum(all_rewards_min) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers_min) / iterations)
            + " | " + "Service Rate:" + str(sum(all_services_min) / iterations) +  " | " + "Upload:" + str(sum(all_upload_min) / iterations) + "\n")

    f.write("**Reward Max:" + str(sum(all_rewards_max) / iterations) + " | " + "Total Energy:" + str(
        sum(all_powers_max) / iterations)
            + " | " + "Service Rate:" + str(sum(all_services_max) / iterations) +  " | " + "Upload:" + str(sum(all_upload_max) / iterations) + "\n")

    f.write("____________________________________________________" + "\n")
    f.close()

    plot_rewards.append(sum(all_rewards) / iterations)
    plot_powers.append(sum(all_powers) / iterations)
    plot_services.append(sum(all_services) / iterations)
    plot_upload.append(sum(all_upload) / iterations)

    plot_rewards_greedy.append(sum(all_rewards_greedy) / iterations)
    plot_powers_greedy.append(sum(all_powers_greedy) / iterations)
    plot_services_greedy.append(sum(all_services_greedy) / iterations)
    plot_upload_greedy.append(sum(all_upload_greedy) / iterations)

    plot_rewards_random.append(sum(all_rewards_random) / iterations)
    plot_powers_random.append(sum(all_powers_random) / iterations)
    plot_services_random.append(sum(all_services_random) / iterations)
    plot_upload_random.append(sum(all_upload_random) / iterations)

    plot_rewards_most.append(sum(all_rewards_most) / iterations)
    plot_powers_most.append(sum(all_powers_most) / iterations)
    plot_services_most.append(sum(all_services_most) / iterations)
    plot_upload_most.append(sum(all_upload_most) / iterations)

    plot_rewards_min.append(sum(all_rewards_min) / iterations)
    plot_powers_min.append(sum(all_powers_min) / iterations)
    plot_services_min.append(sum(all_services_min) / iterations)
    plot_upload_min.append(sum(all_upload_min) / iterations)

    plot_rewards_max.append(sum(all_rewards_max) / iterations)
    plot_powers_max.append(sum(all_powers_max) / iterations)
    plot_services_max.append(sum(all_services_max) / iterations)
    plot_upload_max.append(sum(all_upload_max) / iterations)


plotting.plot_episode_stats(stats, smoothing_window=100)

f = open("result.txt", "a")
f.write("r1="+str(plot_rewards) + "\n")
f.write("c1="+str(plot_powers) + "\n")
f.write("s1="+str(plot_services) + "\n\n")
f.write("u1="+str(plot_upload) + "\n\n")

f.write("r2="+str(plot_rewards_greedy) + "\n")
f.write("c2="+str(plot_powers_greedy) + "\n")
f.write("s2="+str(plot_services_greedy) + "\n\n")
f.write("u2="+str(plot_upload_greedy) + "\n\n")

f.write("r3="+str(plot_rewards_random) + "\n")
f.write("c3="+str(plot_powers_random) + "\n")
f.write("s3="+str(plot_services_random) + "\n\n")
f.write("u3="+str(plot_upload_random) + "\n\n")

f.write("r4="+str(plot_rewards_most) + "\n")
f.write("c4="+str(plot_powers_most) + "\n")
f.write("s4="+str(plot_services_most) + "\n\n")
f.write("u4="+str(plot_upload_most) + "\n\n")

f.write("r5="+str(plot_rewards_min) + "\n")
f.write("c5="+str(plot_powers_min) + "\n")
f.write("s5="+str(plot_services_min) + "\n\n")
f.write("u5="+str(plot_upload_min) + "\n\n")

f.write("r6="+str(plot_rewards_max) + "\n")
f.write("c6="+str(plot_powers_max) + "\n")
f.write("s6="+str(plot_services_max) + "\n")
f.write("u6="+str(plot_upload_max) + "\n\n")

f.write("________________________END EXPERMINT____________________________" + "\n")
