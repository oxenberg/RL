from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
from tensorflow import concat
from tensorflow.compat.v1 import Session
from tensorflow.compat.v1 import placeholder
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.python.ops.distributions.normal import Normal
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2
from tensorflow.python.ops.variable_scope import variable_scope
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.saver import Saver

from HW3.actor_critic import ACTION_SIZE
from HW3.actor_critic import ENV_TO_ACTION_SIZE
from HW3.actor_critic import ENV_TO_REWARD_THRESHOLD
from HW3.actor_critic import ENV_TO_STATE_SIZE
from HW3.actor_critic import OpenGymEnvs
from HW3.actor_critic import PolicyNetwork
from HW3.actor_critic import STATE_SIZE
from HW3.actor_critic import ValueNetwork

MAX_EPISODES = 2500


class ProgressivePolicyNetwork:
    def __init__(self,
                 learning_rate,
                 policy_source1: PolicyNetwork,
                 policy_source2: PolicyNetwork,
                 policy_out: PolicyNetwork,
                 mountain_car=None,
                 name='progressive_policy_network'):
        self.learning_rate = learning_rate
        self.name = name
        self.init = tf.initializers.GlorotUniform()

        with variable_scope(self.name):
            self.action = placeholder(tf.float32, [ACTION_SIZE], name="action")
            self.R_t = placeholder(tf.float32, name="total_rewards")
            self.reward_per_episode = placeholder(tf.float32, name="reware_per_episode")
            tf.compat.v1.summary.scalar('rewards', self.reward_per_episode)

            ### Output network ###
            A2_3_input = concat([policy_source1.A1, policy_source2.A1, policy_out.A1], axis=1)
            concatenated_W2_3 = concat([policy_source1.W2, policy_source2.W2, policy_out.W2], axis=0)
            concatenated_b2_3 = concat([policy_source1.b2, policy_source2.b2, policy_out.b2], axis=0)

            Z2_1 = tf.add(tf.matmul(policy_source1.A1, policy_source1.W2), policy_source1.b2)
            Z2_2 = tf.add(tf.matmul(policy_source2.A1, policy_source2.W2), policy_source2.b2)
            Z2_3 = tf.add(tf.matmul(policy_out.A1, policy_out.W2), policy_out.b2)

            if mountain_car:
                self.output_mu = tf.add(tf.matmul(A2_3_input, concatenated_W2_3), concatenated_b2_3)
                self.output_var = tf.add(tf.matmul(A2_3_input, concatenated_W2_3), concatenated_b2_3)

                self.output_mu = tf.squeeze(self.output_mu)
                self.output_var = tf.squeeze(self.output_var)
                self.output_var = tf.nn.softplus(self.output_var) + 1e-5
                self.normal_dist = Normal(self.output_mu, self.output_var)
                self.sampled_action = tf.squeeze(
                    self.normal_dist._sample_n(1), axis=0)
                self.actions_distribution = tf.clip_by_value(self.sampled_action, -1, 1)
                # Loss and train op

                self.loss = - tf.math.log(self.normal_dist.prob(
                    tf.squeeze(self.action)) + 1e-5) * self.R_t
                # Add cross entropy cost to encourage exploration
                # self.loss -= 1e-1 * self.normal_dist.entropy()
            else:
                self.output = tf.add(tf.matmul(A2_3_input, concatenated_W2_3), concatenated_b2_3)

                # Softmax probability distribution over actions
                self.actions_distribution = tf.squeeze(
                    tf.nn.softmax(self.output))
                # Loss with negative log probability
                self.neg_log_prob = softmax_cross_entropy_with_logits_v2(
                    logits=self.output, labels=self.actions_distribution)
                self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)

            self.vars_to_train = [policy_out.W1, policy_out.b1,
                                  policy_out.W2, policy_out.b2]
            self.optimizer = AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss, var_list=self.vars_to_train)
            self.merged = tf.compat.v1.summary.merge_all()


class ProgressiveAgent:
    def __init__(self, env_name, source_dict):
        self.env_name = source_dict["out"]
        self.source_dict = source_dict
        self.env = gym.make(self.env_name)
        self.convergence_treshold = ENV_TO_REWARD_THRESHOLD[env_name]
        self.original_action_size = ENV_TO_ACTION_SIZE[env_name]
        self.original_state_size = ENV_TO_STATE_SIZE[env_name]

    def run(self, discount_factor, learning_rate_policy, learning_rate_value,
            num_hidden_layers, num_neurons_value, num_neurons_policy):
        # Initialize the policy network
        reset_default_graph()

        mountain_car = self.env_name == OpenGymEnvs.MOUNTAIN_CAR.value

        self.policy_source1 = PolicyNetwork(
            learning_rate_policy, num_neurons_policy, mountain_car=mountain_car,
            progressive=self.source_dict["input_1"])
        self.policy_source2 = PolicyNetwork(
            learning_rate_policy, num_neurons_policy, mountain_car=mountain_car,
            progressive=self.source_dict["input_2"])

        self.policy_out = PolicyNetwork(
            learning_rate_policy, num_neurons_policy, mountain_car=mountain_car,
            progressive=self.source_dict["out"])

        self.progressive_policy = ProgressivePolicyNetwork(learning_rate_policy,
                                                           self.policy_source1,
                                                           self.policy_source2,
                                                           self.policy_out,
                                                           mountain_car=mountain_car)

        self.value_function = ValueNetwork(learning_rate_value, num_hidden_layers, num_neurons_value)

        saver = Saver(var_list=self.policy_source1.var_to_save + self.policy_source2.var_to_save)

        with Session() as sess:
            sess.run(global_variables_initializer())

            saver.restore(sess, f"/tmp/{self.source_dict['input_1']}-for_transfer-model.ckpt")
            saver.restore(sess, f"/tmp/{self.source_dict['input_2']}-for_transfer-model.ckpt")

            # initiate log files
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = (f'../logs/gradient_tape/{self.env_name}/' +
                             current_time + '/train')
            train_summary_writer = tf.compat.v1.summary.FileWriter(train_log_dir, sess.graph)

            all_avg = []
            solved = False
            episode_rewards = np.zeros(MAX_EPISODES)
            average_rewards = -np.inf

            for episode in range(MAX_EPISODES):
                current_state = self.env.reset()
                current_state = np.reshape(np.pad(current_state, (0, STATE_SIZE - self.original_state_size)),
                                           [1, STATE_SIZE])
                I = 1
                while True:
                    actions_distribution = sess.run(self.progressive_policy.actions_distribution,
                                                    {self.policy_source1.state: current_state,
                                                     self.policy_source2.state: current_state,
                                                     self.policy_out.state: current_state})
                    if not mountain_car:
                        actions_distribution = actions_distribution[:self.original_action_size]
                        actions_distribution = actions_distribution / actions_distribution.sum()
                        action = np.random.choice(
                            np.arange(self.original_action_size), p=actions_distribution)
                    else:
                        action = [actions_distribution[0]]
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(np.pad(next_state, (0, STATE_SIZE - self.original_state_size)),
                                            [1, STATE_SIZE])

                    episode_rewards[episode] += reward

                    current_state_prediction = sess.run([self.value_function.final_output],
                                                        {self.value_function.state: current_state,
                                                         self.value_function.total_discounted_return: reward})
                    next_state_prediction = sess.run([self.value_function.final_output],
                                                     {self.value_function.state: next_state,
                                                      self.value_function.total_discounted_return: reward})

                    reward_estemation = reward + \
                                        (1 - int(done)) * discount_factor * \
                                        next_state_prediction[0][0][0]
                    delta = reward_estemation - \
                            current_state_prediction[0][0][0]
                    delta *= I

                    sess.run([self.value_function.optimizer], {self.value_function.state: current_state,
                                                               self.value_function.total_discounted_return: reward_estemation,
                                                               self.value_function.delta: 1})

                    action_one_hot = np.zeros(ACTION_SIZE)
                    if mountain_car:
                        action_one_hot[0] = action[0]
                    else:
                        action_one_hot[action] = 1
                    feed_dict = {self.policy_source1.state: current_state,
                                 self.policy_source2.state: current_state,
                                 self.policy_out.state: current_state,
                                 self.progressive_policy.R_t: delta,
                                 self.progressive_policy.action: action_one_hot,
                                 self.progressive_policy.reward_per_episode: episode_rewards[episode]}
                    _, summary = sess.run([self.progressive_policy.optimizer,
                                           self.progressive_policy.merged],
                                          feed_dict)

                    if done:
                        if episode > 98:
                            # Check if solved
                            average_rewards = np.mean(
                                episode_rewards[(episode - 99):episode + 1])
                        print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode,
                                                                                           episode_rewards[episode],
                                                                                           round(average_rewards, 2)))
                        all_avg.append(round(average_rewards, 2))
                        if average_rewards >= self.convergence_treshold:
                            print(' Solved at episode: ' + str(episode))
                            solved = True
                        break

                    current_state = next_state
                    I *= discount_factor

                if solved:
                    break

                train_summary_writer.add_summary(summary, episode)

        return max(all_avg)


if __name__ == '__main__':
    transfer_dict = {'input_1': OpenGymEnvs.CARTPOLE.value,
                     'input_2': OpenGymEnvs.ACROBOT.value,
                     'out': OpenGymEnvs.MOUNTAIN_CAR.value}
    agent = ProgressiveAgent(OpenGymEnvs.MOUNTAIN_CAR, transfer_dict)
    parameters = {'discount_factor': 0.99, 'learning_rate_policy': 0.00001, 'learning_rate_value': 0.0004,
                  'num_hidden_layers': 2, 'num_neurons_value': 12, 'num_neurons_policy': 12}
    agent.run(**parameters)
