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

env = gym.make('CartPole-v1')

np.random.seed(1)
tf.compat.v1.disable_eager_execution()

USE_BASELINE = False

## Define hyperparameters
state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501

# policy
discount_factor = 0.5
learning_rate = 0.0004

# ValueFunction
discount_factor_value = 0.99
learning_rate_value = 0.0004
num_hidden_layers = 3
num_neurons = 24

render = False

## Initialize the policy network
reset_default_graph()
policy = PolicyNetwork(state_size, action_size, learning_rate)
value_function = ValueNetwork(state_size, learning_rate_value, num_hidden_layers, num_neurons)

# Start training the agent with REINFORCE algorithm

print("Dsdsdsdsdsdsdsd")

with Session() as sess:
    sess.run(global_variables_initializer())

    # initiate log files
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S") + ('-baseline' if USE_BASELINE else '')
    train_log_dir = '../logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.compat.v1.summary.FileWriter(train_log_dir, sess.graph)

    solved = False
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
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

            delta = reward + discount_factor * next_state_prediction[0] - current_state_prediction[0]
            sess.run([value_function.optimizer], {value_function.state: current_state,
                                                  value_function.total_discounted_return: delta})

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
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode,
                                                                                   episode_rewards[episode],
                                                                                   round(average_rewards, 2)))
                if average_rewards > 475:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break

            current_state = next_state
            I *= discount_factor

        if solved:
            break

        train_summary_writer.add_summary(summary, episode)  # ToDo: check this
