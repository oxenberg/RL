from datetime import datetime
from enum import Enum
from random import shuffle

import gym
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from tensorflow.compat.v1 import Session
from tensorflow.compat.v1 import placeholder
from tensorflow.compat.v1.losses import mean_squared_error
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.python.ops.distributions.normal import Normal
# v2 tf embaded
from tensorflow.python.ops.init_ops import GlorotNormal
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2
from tensorflow.python.ops.variable_scope import get_variable
from tensorflow.python.ops.variable_scope import variable_scope
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.training.adam import AdamOptimizer
# v2 tf embaded
from tensorflow.python.training.saver import Saver
from tqdm import tqdm

# np.random.seed(1)
tf.compat.v1.disable_eager_execution()

STATE_SIZE = 6
ACTION_SIZE = 3


class OpenGymEnvs(Enum):
    CARTPOLE = 'CartPole-v1'
    ACROBOT = 'Acrobot-v1'
    MOUNTAIN_CAR = 'MountainCarContinuous-v0'


ENV_TO_REWARD_THRESHOLD = {
    OpenGymEnvs.CARTPOLE: 475,
    OpenGymEnvs.ACROBOT: -100,
    OpenGymEnvs.MOUNTAIN_CAR: 90
}

ENV_TO_ACTION_SIZE = {
    OpenGymEnvs.CARTPOLE: 2,
    OpenGymEnvs.ACROBOT: 3,
    OpenGymEnvs.MOUNTAIN_CAR: 1
}

ENV_TO_STATE_SIZE = {
    OpenGymEnvs.CARTPOLE: 4,
    OpenGymEnvs.ACROBOT: 6,
    OpenGymEnvs.MOUNTAIN_CAR: 2
}

MAX_EPISODES = 2500
RENDER = False


class PolicyNetwork:
    def __init__(self, learning_rate, neurons=12, name='policy_network', retrain=False, mountain_car=False,
                 progressive=None):
        self.learning_rate = learning_rate
        self.name = name
        self.init = tf.initializers.GlorotNormal()
        self.prefix_var = ""
        if progressive:
            self.prefix_var = progressive
        with variable_scope(self.name):
            self.state = placeholder(
                tf.float32, [None, STATE_SIZE], name="state")
            self.action = placeholder(tf.float32, [ACTION_SIZE], name="action")
            self.R_t = placeholder(tf.float32, name="total_rewards")
            self.reward_per_episode = placeholder(
                tf.float32, name="reware_per_episode")
            tf.compat.v1.summary.scalar('rewards', self.reward_per_episode)

            self.W1 = get_variable(
                f"{self.prefix_var}W1", [STATE_SIZE, neurons], initializer=self.init)
            self.b1 = get_variable(
                f"{self.prefix_var}b1", [neurons], initializer=tf.zeros_initializer())
            self.W2 = get_variable(
                f"{self.prefix_var}W2", [neurons, ACTION_SIZE], initializer=self.init)
            self.b2 = get_variable(
                f"{self.prefix_var}b2", [ACTION_SIZE], initializer=tf.zeros_initializer())
            if retrain:
                self.W2 = get_variable(f"{self.prefix_var}W2_retrain", [neurons, ACTION_SIZE],
                                       initializer=GlorotNormal(seed=0))
                self.b2 = get_variable(f"{self.prefix_var}b2_retrain", [ACTION_SIZE],
                                       initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            if mountain_car:
                self.output_mu = tf.add(tf.matmul(self.A1, self.W2), self.b2)
                self.output_var = tf.add(tf.matmul(self.A1, self.W2), self.b2)

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
                self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

                # Softmax probability distribution over actions
                self.actions_distribution = tf.squeeze(
                    tf.nn.softmax(self.output))
                # Loss with negative log probability
                self.neg_log_prob = softmax_cross_entropy_with_logits_v2(
                    logits=self.output, labels=self.action)
                self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)
            self.merged = tf.compat.v1.summary.merge_all()

            self.var_to_save = [self.W1, self.b1]


class ValueNetwork:
    def __init__(self, learning_rate, num_hidden_layers, num_neurons, name='value_network'):
        self.learning_rate = learning_rate
        self.num_neurons = num_neurons
        self.num_hidden_layers = num_hidden_layers
        with variable_scope(name):
            self.state = placeholder(
                tf.float32, [None, STATE_SIZE], name=f"{name}_state")
            self.total_discounted_return = placeholder(
                tf.float32, name=f"{name}_total_discounted_return")
            self.delta = placeholder(tf.float32, name=f"{name}_delta")
            self.var_to_save = []

            W1 = get_variable(
                f"{name}_W1", [STATE_SIZE, self.num_neurons], initializer=GlorotNormal(seed=0))
            b1 = get_variable(
                f"{name}_b1", [self.num_neurons], initializer=tf.zeros_initializer())
            self.var_to_save.extend([W1, b1])
            Z1 = tf.add(tf.matmul(self.state, W1), b1)
            A = tf.nn.relu(Z1)

            for i in range(2, self.num_hidden_layers + 1):
                W = get_variable(f"{name}_W{i}", [
                    self.num_neurons, self.num_neurons], initializer=GlorotNormal(seed=0))
                b = get_variable(f"{name}_b{i}", [
                    self.num_neurons], initializer=tf.zeros_initializer())
                self.var_to_save.extend([W, b])
                Z = tf.add(tf.matmul(A, W), b)
                A = tf.nn.relu(Z)

            W = get_variable(f"{name}_W{self.num_hidden_layers + 1}", [self.num_neurons, 1],
                             initializer=GlorotNormal(seed=0))
            b = get_variable(f"{name}_b{self.num_hidden_layers + 1}",
                             [1], initializer=tf.zeros_initializer())
            # linear activation function
            self.final_output = tf.add(
                tf.matmul(A, W), b, name=f"{name}_final_output")
            # Softmax probability distribution over actions
            self.loss = mean_squared_error(
                self.total_discounted_return, self.final_output)
            self.loss *= self.delta
            self.optimizer = AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)


# function to normalize states
def scale_state(scaler, state):  # requires input shape=(2,)
    scaled = scaler.transform(state)
    return scaled


class Agent:
    def __init__(self, env_name):
        self.env_name = env_name.value

        self.env = gym.make(self.env_name)
        self.convergence_threshold = ENV_TO_REWARD_THRESHOLD[env_name]
        self.original_action_size = ENV_TO_ACTION_SIZE[env_name]
        self.original_state_size = ENV_TO_STATE_SIZE[env_name]

    def _pad_state(self, state):
        return np.reshape(np.pad(state, (0, STATE_SIZE - self.original_state_size)),
                          [1, STATE_SIZE])

    def run(self, discount_factor, learning_rate, learning_rate_value,
            num_hidden_layers, num_neurons_value, num_neurons_policy, restore_sess=None, for_transfer=False):
        # Initialize the policy network
        reset_default_graph()

        mountain_car = self.env_name == OpenGymEnvs.MOUNTAIN_CAR.value

        if mountain_car:
            state_space_samples = np.array(
                [self._pad_state(self.env.observation_space.sample())
                 for x in range(10000)]).reshape(10000, STATE_SIZE)
            scaler = StandardScaler()
            scaler.fit(state_space_samples)

        self.policy = PolicyNetwork(
            learning_rate, num_neurons_policy, retrain=restore_sess is not None, mountain_car=mountain_car)
        self.value_function = ValueNetwork(
            learning_rate_value, num_hidden_layers, num_neurons_value)

        saver = Saver(var_list=self.policy.var_to_save)

        with Session() as sess:
            sess.run(global_variables_initializer())

            if restore_sess:
                saver.restore(sess, f"/tmp/{restore_sess}-model.ckpt")

            # initiate log files
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = f'../logs/gradient_tape/{self.env_name}/' + \
                            current_time + '/train'
            train_summary_writer = tf.compat.v1.summary.FileWriter(
                train_log_dir, sess.graph)

            all_avg = []
            solved = False
            episode_rewards = np.zeros(MAX_EPISODES)
            average_rewards = -np.inf

            for episode in range(MAX_EPISODES):
                current_state = self.env.reset()
                current_state = self._pad_state(current_state)
                if mountain_car:
                    current_state = scale_state(scaler, current_state)

                I = 1
                while True:
                    actions_distribution = sess.run(self.policy.actions_distribution,
                                                    {self.policy.state: current_state})
                    if mountain_car:
                        action = [actions_distribution[0]]
                    else:
                        actions_distribution = actions_distribution[:self.original_action_size]
                        actions_distribution = actions_distribution / actions_distribution.sum()
                        action = np.random.choice(
                            np.arange(self.original_action_size), p=actions_distribution)

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = self._pad_state(next_state)
                    if mountain_car:
                        next_state = scale_state(scaler, next_state)

                    episode_rewards[episode] += reward

                    if RENDER:
                        self.env.render()

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
                    feed_dict = {self.policy.state: current_state,
                                 self.policy.R_t: delta,
                                 self.policy.action: action_one_hot,
                                 self.policy.reward_per_episode: episode_rewards[episode]}
                    _, loss, summary = sess.run([self.policy.optimizer,
                                                 self.policy.loss,
                                                 self.policy.merged],
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
                        if average_rewards >= self.convergence_threshold:
                            print(' Solved at episode: ' + str(episode))
                            solved = True
                        break

                    current_state = next_state
                    I *= discount_factor

                if solved:
                    break

                train_summary_writer.add_summary(summary, episode)

            transfered = '-transfered' if restore_sess else ''
            for_transfer = '-for-transfer' if for_transfer else ''
            save_path = saver.save(
                sess, f"/tmp/{self.env_name + transfered + for_transfer}-model.ckpt")
            print("Model saved in path: %s" % save_path)
        return max(all_avg)


def gridSearch(parmas, env_name, nSearch=100, maxN=False):
    paramsList = list(ParameterGrid(parmas))
    shuffle(paramsList)

    #     if nSearch > len(paramsList) or maxN:
    #         nSearch = len(paramsList)

    gridSearchResults = []
    agent = Agent(env_name)
    for paramsDict in tqdm(paramsList[:nSearch]):
        try:
            print(paramsDict)
            max_reward_avg = agent.run(**paramsDict)
            paramsDict['max_average_reward_100_episodes'] = max_reward_avg
            print(paramsDict)
            gridSearchResults.append(paramsDict)
        except Exception as e:
            print(e)
            print(paramsDict)
            continue

    hyperparameterTable = pd.DataFrame(gridSearchResults)
    hyperparameterTable.sort_values(
        "max_average_reward_100_episodes", inplace=True)
    hyperparameterTable.to_csv(f"HP-{env_name.value}.csv")
    print(hyperparameterTable)


def run_grid_search(env):
    params = {"discount_factor": [0.99, 0.9, 0.95],
              "learning_rate": [0.01, 0.001, 0.0001, 0.00001],
              "learning_rate_value": [0.01, 0.001, 0.0001, 0.00001],
              "num_hidden_layers": [2, 3, 5],
              "num_neurons_value": [8, 16, 32, 64],
              "num_neurons_policy": [8, 16, 32, 64]}

    gridSearch(params, env)


def run_grid_search_transfer(env, transfer, params):
    params = {"discount_factor": [0.99, 0.9, 0.95],
              "learning_rate": [0.01, 0.001, 0.0001, 0.00001],
              "learning_rate_value": [0.01, 0.001, 0.0001, 0.00001],
              **params,
              'restore_sess': [transfer]}
    gridSearch(params, env)


if __name__ == '__main__':
    agent = Agent(OpenGymEnvs.MOUNTAIN_CAR)
    best_parameters = {'discount_factor': 0.99, 'learning_rate': 0.00002, 'learning_rate_value': 0.001,
                       'num_hidden_layers': 2, 'num_neurons_value': 400, 'num_neurons_policy': 40}

    # run_grid_search(OpenGymEnvs.MOUNTAIN_CAR)
    ret = agent.run(**best_parameters)
    # ret = agent.run(**best_parameters,restore_sess = True)
