from myenv import MyEnv
import numpy as np
import plotting
import matplotlib.pyplot as plt
import pandas as pd
import random

from DQN_brain import DeepQNetwork

if __name__ == "__main__":

    densities = [0.002]  # 0.002, 0.006, 0.01
    smooth = 3
    plot_DQN_rewards = []
    plot_DQN_serveratio = []
    plot_DQN_incentives = []

    plot_RANDOM_rewards = []
    plot_RANDOM_serveratio = []
    plot_RANDOM_incentives = []

    plot_POPULAR_rewards = []
    plot_POPULAR_serveratio = []
    plot_POPULAR_incentives = []

    plot_MIN_rewards = []
    plot_MIN_serveratio = []
    plot_MIN_incentives = []

    RL = False
    for density in densities:
        DQN_rewards = []
        DQN_serveratio = []
        DQN_incentives = []

        RANDOM_rewards = []
        RANDOM_serveratio = []
        RANDOM_incentives = []

        POPULAR_rewards = []
        POPULAR_serveratio = []
        POPULAR_incentives = []

        MIN_rewards = []
        MIN_serveratio = []
        MIN_incentives = []

        rewards = []

        # Number of trials (episodes)
        no_episodes = 50

        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(no_episodes),
            episode_rewards=np.zeros(no_episodes))

        T = 2000
        number_of_contents = 10
        myenv = MyEnv(density=density, T=T, number_of_contents=number_of_contents)

        if (RL == False):
            RL = DeepQNetwork(myenv.no_actions, myenv.observation_length,
                              learning_rate=0.001,
                              reward_decay=0.9,
                              e_greedy=0.9,
                              replace_target_iter=5000,
                              memory_size=2000,
                              batch_size=220
                              # output_graph=True
                              )

        print("No. vehicles:" + str(myenv.number_of_vehicles));

        for e in range(no_episodes):

            myenv = MyEnv(density=density, T=T, number_of_contents=number_of_contents)
            myenv.episode = e
            myenv.no_episodes = no_episodes
            # Reset the envirounment
            observation = myenv.reset();
            #print("eeee", e)
            # if (e == no_episodes - 1):
            #    print(myenv.contents_sizes)
            for t in range(T - 1):
                #print(t)
                # take action
                # RL choose action based on observation
                action = RL.choose_action(np.array(observation))

                if (e == no_episodes - 1):
                    print("Observation:", observation)
                    print("Action:", action)

                # process the action and get the new observation
                reward, observation_, Done = myenv.step(action)

                if (e == no_episodes - 1):
                    print("RSU cache:", myenv.RSU_cache)
                    print("Fetched:", myenv.fetched)
                    print("Served:", myenv.served)
                    print("Reward:", reward)
                    print("Total Reward:", myenv.total_reward)
                    print("\n****")

                # Next observation is terminal
                # if (t == myenv.T - 1):
                # observation_ = 'terminal'

                # RL learn from this transition
                RL.store_transition(np.array(observation), action, reward, np.array(observation_))

                if (t % 5 == 0):
                    RL.learn()
                # print(observation)
                # print(action)
                # print(observation_)
                # print("_____________________________________________")

                # swap current and next observation
                observation = observation_

                stats.episode_rewards[e] += reward
                stats.episode_lengths[e] = t

            DQN_rewards.append(myenv.total_reward)
            DQN_serveratio.append(myenv.total_download / myenv.total_request_amount)
            DQN_incentives.append(myenv.total_energy)

            if (e % 30 == 0):
                print("Episode:", str(e));

            if (e == no_episodes - 1):
                print("availables:", myenv.available)
                print("requests:", myenv.requests)
                print("content sizes:", myenv.contents_sizes)
                print("total reward:", myenv.total_reward)
                print("total incentives:", myenv.total_energy)
                print("total download:", myenv.total_download)
            #     ### RANDOM starts
            #     myenv.reset()
            #     myenv.i = 0
            #     taken = 0
            #     for t in range(T):
            #         reward, observation_, Done = myenv.step(random.randint(0,myenv.no_actions-1))
            #     RANDOM_rewards.append(myenv.total_reward)
            #     RANDOM_serveratio.append(myenv.total_download/myenv.total_request_amount)
            #     RANDOM_incentives.append(myenv.total_energy)
            #
            ### POPULAR starts

            myenv.reset()
            commit = -1

            for t in range(T - 1):
                action = myenv.no_actions - 1
                max_index = 0
                for p in range(myenv.no_actions):
                    if (observation_[p * 4] > observation_[max_index * 4]):
                        max_index = p

                prefix = ((myenv.no_actions - 1) * 4) - 1

                cached_min_index = 0

                for cached in range(myenv.max_cached_contents):
                    if (observation_[prefix + (cached * 2)] < observation_[prefix + cached_min_index]):
                        cached_min_index = cached

                if (observation_[max_index * 4] > observation_[prefix + cached_min_index] or myenv.RSU_cache_free() >=
                        observation_[(max_index * 4) + 2]):
                    action = max_index

                reward, observation_, Done = myenv.step(action)
            #     if (e == no_episodes - 1):
            #         print("POPULAR RSU cache:", myenv.RSU_cache)
            #
            if (e == no_episodes - 1):
                print("Populer total reward:", myenv.total_reward)
                print("Populer total incentives:", myenv.total_energy)
                print("Populer total download:", myenv.total_download)

            POPULAR_rewards.append(myenv.total_reward)
            POPULAR_serveratio.append(myenv.total_download / myenv.total_request_amount)
            POPULAR_incentives.append(myenv.total_energy)
        #
        #     ### MIN starts
        #     myenv.reset()
        #     myenv.i = 0
        #     taken = 0
        #     commit = -1
        #     for t in range(T):
        #         action = myenv.no_actions-1
        #         max_index = 0
        #         for p in range(myenv.no_actions):
        #             if(observation_[1+p*3] < observation_[1+max_index * 3]):
        #                 max_index = p
        #
        #         reward, observation_, Done = myenv.step(max_index)
        #         #if (e == no_episodes - 1):
        #             #print("POPULAR RSU cache:", myenv.RSU_cache)
        #
        #     MIN_rewards.append(myenv.total_reward)
        #     MIN_serveratio.append(myenv.total_download / myenv.total_request_amount)
        #     MIN_incentives.append(myenv.total_energy)
        #
        # plot_DQN_rewards.append(sum(DQN_rewards)/no_episodes)
        # plot_DQN_serveratio.append(sum(DQN_serveratio)/no_episodes)
        # plot_DQN_incentives.append(sum(DQN_incentives)/no_episodes)
        #
        # plot_RANDOM_rewards.append(sum(RANDOM_rewards) / no_episodes)
        # plot_RANDOM_serveratio.append(sum(RANDOM_serveratio)/no_episodes)
        # plot_RANDOM_incentives.append(sum(RANDOM_incentives)/no_episodes)
        #
        plot_POPULAR_rewards.append(sum(POPULAR_rewards) / no_episodes)
        plot_POPULAR_serveratio.append(sum(POPULAR_serveratio) / no_episodes)
        plot_POPULAR_incentives.append(sum(POPULAR_incentives) / no_episodes)
        #
        # plot_MIN_rewards.append(sum(MIN_rewards) / no_episodes)
        # plot_MIN_serveratio.append(sum(MIN_serveratio) / no_episodes)
        # plot_MIN_incentives.append(sum(MIN_incentives) / no_episodes)

    print("DQN Reward:" + str(sum(DQN_rewards) / no_episodes) + " Serve Ratio:" + str(sum(DQN_serveratio) / no_episodes)
          + " Incentives:" + str(sum(DQN_incentives) / no_episodes))
    # print("RANDOM Reward:" + str(sum(RANDOM_rewards) / no_episodes) + " Serve Ratio:"+str(sum(RANDOM_serveratio)/no_episodes)
    #       + " Incentives:"+str(sum(RANDOM_incentives)/no_episodes))
    print("POPULAR Reward:" + str(sum(POPULAR_rewards) / no_episodes) + " Serve Ratio:" + str(
        sum(POPULAR_serveratio) / no_episodes)
          + " Incentives:" + str(sum(POPULAR_incentives) / no_episodes))
    # print("MIN Reward:" + str(sum(MIN_rewards) / no_episodes) + " Serve Ratio:"+str(sum(MIN_serveratio)/no_episodes)
    #       + " Incentives:"+str(sum(MIN_incentives)/no_episodes))

    # plotting.plot_episode_stats(stats, smoothing_window=200)

    rewards_smoothed = pd.Series(DQN_rewards).rolling(smooth, min_periods=smooth).mean()
    plt.plot(rewards_smoothed, linewidth=5, color='g', label='DQN')

    rewards_smoothed = pd.Series(RANDOM_rewards).rolling(smooth, min_periods=smooth).mean()
    plt.plot(rewards_smoothed, linewidth=5, color='r', label='RANDOM')

    rewards_smoothed = pd.Series(POPULAR_rewards).rolling(smooth, min_periods=smooth).mean()
    plt.plot(rewards_smoothed, linewidth=5, color='b', label='POPULAR')

    rewards_smoothed = pd.Series(MIN_rewards).rolling(smooth, min_periods=smooth).mean()
    plt.plot(rewards_smoothed, linewidth=5, color='y', label='MIN')

    plt.ylabel('Total Revenue (X100 Units)', fontsize=40)
    plt.xlabel('Episodes', fontsize=40)
    plt.legend(loc='upper left')
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.legend(prop={'size': 33})
    plt.grid()
    plt.show()

    f = open("result.txt", "a")
    f.write("r1=" + str(plot_DQN_rewards) + "\n")
    f.write("c1=" + str(plot_DQN_serveratio) + "\n")
    f.write("s1=" + str(plot_DQN_incentives) + "\n\n")

    f.write("r2=" + str(plot_RANDOM_rewards) + "\n")
    f.write("c2=" + str(plot_RANDOM_serveratio) + "\n")
    f.write("s2=" + str(plot_RANDOM_incentives) + "\n\n")

    f.write("r3=" + str(plot_POPULAR_rewards) + "\n")
    f.write("c3=" + str(plot_POPULAR_serveratio) + "\n")
    f.write("s3=" + str(plot_POPULAR_incentives) + "\n\n")

    f.write("r4=" + str(plot_MIN_rewards) + "\n")
    f.write("c4=" + str(plot_MIN_serveratio) + "\n")
    f.write("s4=" + str(plot_MIN_incentives) + "\n\n")

    f.write("________________________END EXPERMINT____________________________" + "\n")
