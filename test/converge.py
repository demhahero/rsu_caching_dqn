from DQN_brain import DeepQNetwork
from myenv import MyEnv
import numpy as np
import plotting
import matplotlib.style

matplotlib.style.use('ggplot')

if __name__ == "__main__":

    densities = [0.002]
    for density in densities:

        rewards = []

        # Number of trials (episodes)
        no_episodes = 4000;

        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(no_episodes),
            episode_rewards=np.zeros(no_episodes))


        myenv = MyEnv(density=density)
        print(myenv.number_of_vehicles)
        RL = DeepQNetwork(myenv.no_actions, myenv.number_of_contents+2,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=100,
                      # output_graph=True
                      )

        for e in range(no_episodes):

            # Reset the envirounment
            observation = myenv.reset();

            for i in range(myenv.number_of_vehicles):

                # take action
                # RL choose action based on observation
                action = RL.choose_action(np.array(observation))

                # process the action and get the new observation
                reward, observation_, Done = myenv.step(action)

                # Next observation is terminal
                #if (t == myenv.T - 1):
                    #observation_ = 'terminal'

                # RL learn from this transition
                RL.store_transition(np.array(observation), action, reward, np.array(observation_))

                if (i > 20) and (i % 5 == 0):
                    RL.learn()
                # print(observation)
                # print(action)
                # print(observation_)
                # print("_____________________________________________")

                # swap current and next observation
                observation = observation_

                stats.episode_rewards[e] += reward
                stats.episode_lengths[e] = i

            rewards.append(myenv.total_reward)

    plotting.plot_episode_stats(stats)
