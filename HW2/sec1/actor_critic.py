import collections
from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import Session
from tensorflow.python.framework.ops import reset_default_graph
##v2 tf embaded
from tensorflow.python.ops.variables import global_variables_initializer

from policy_gradients import PolicyNetwork
from policy_gradients import ValueNetwork
from random import shuffle
from tqdm import tqdm
import pandas as pd

from sklearn.model_selection import ParameterGrid


env = gym.make('CartPole-v1')

np.random.seed(1)
tf.compat.v1.disable_eager_execution()

USE_BASELINE = False
state_size = 4
max_episodes = 5000
max_steps = 501
action_size = env.action_space.n

render = False




def gridSearch(parmas, nSearch=1, maxN=False, aveOver=15):
    paramsList = list(ParameterGrid(parmas))
    shuffle(paramsList)

    if nSearch > len(paramsList) or maxN:
        nSearch = len(paramsList)

    gridSearchResults = []
    for paramsDict in tqdm(paramsList[:nSearch]):
        try:
            max_reward_avg = run(**paramsDict)
            paramsDict['max_average_reward_100_episodes'] = max_reward_avg
            print(paramsDict)
            gridSearchResults.append(paramsDict)
        except Exception as e:
            print(e)
            continue

    hyperparameterTable = pd.DataFrame(gridSearchResults)
    hyperparameterTable.sort_values("max_average_reward_100_episodes", inplace=True)
    hyperparameterTable.to_csv("Part2-HP.csv")
    print(hyperparameterTable)



# Start training the agent with REINFORCE algorithm
def run(discount_factor,learning_rate,learning_rate_value,num_hidden_layers,num_neurons):
    
    ## Define hyperparameters
    
    # policy
    discount_factor = 0.99
    learning_rate = 0.0004
    
    # ValueFunction
    learning_rate_value = 0.0004
    num_hidden_layers = 3
    num_neurons = 24
    
    ## Initialize the policy network
    reset_default_graph()
    policy = PolicyNetwork(state_size, action_size, learning_rate)
    value_function = ValueNetwork(state_size, learning_rate_value, num_hidden_layers, num_neurons)
    with Session() as sess:
        sess.run(global_variables_initializer())
    
        # initiate log files
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S") + ('-baseline' if USE_BASELINE else '')
        train_log_dir = '../logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.compat.v1.summary.FileWriter(train_log_dir, sess.graph)
        
        all_avg = []
        solved = False
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0
    
        for episode in range(max_episodes):
            current_state = env.reset()
            current_state = current_state.reshape([1, state_size])
            episode_transitions = []
            I = 1
            for step in range(max_steps):
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: current_state})
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])
                
                episode_rewards[episode] += reward
                
                if render:
                    env.render()
    
                current_state_prediction = sess.run([value_function.final_output],
                                                    {value_function.state: current_state,
                                                     value_function.total_discounted_return: reward})
                next_state_prediction = sess.run([value_function.final_output],
                                                 {value_function.state: next_state,
                                                  value_function.total_discounted_return: reward})
                
                reward_estemation = reward + (1- int(done))*discount_factor * next_state_prediction[0][0][0]
                delta = reward_estemation - current_state_prediction[0][0][0]
                delta *= I
                
                sess.run([value_function.optimizer], {value_function.state: current_state,
                                                      value_function.total_discounted_return: reward_estemation,
                                                      value_function.delta : 1})
    
                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                feed_dict = {policy.state: current_state,
                             policy.R_t: delta,
                             policy.action: action_one_hot,
                             policy.reward_per_episode: episode_rewards[episode]}  # ToDo: fix this
                _, _, loss, summary = sess.run([policy.output,
                                                policy.optimizer,
                                                policy.loss,
                                                policy.merged],
                                               feed_dict)
    
                
                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])  # ToDo: Check this
                    # print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode,
                    #                                                                    episode_rewards[episode],
                    #                                                                    round(average_rewards, 2)))
                    all_avg.append(round(average_rewards, 2))
                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                    break
    
                current_state = next_state
                I *= discount_factor
    
            if solved:
                break
    
            train_summary_writer.add_summary(summary, episode)  # ToDo: check this
    return max(all_avg)


params = {"discount_factor": np.arange(0.9,1,0.1),
          "learning_rate":np.arange(0.0001,0.01,0.0003),
          "learning_rate_value": np.arange(0.0001,0.01,0.0003),
          "num_hidden_layers": np.arange(2,6,1),
          "num_neurons":np.arange(6,32,2)}

gridSearch(params)


