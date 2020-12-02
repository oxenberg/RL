import collections
from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
from keras.losses import MSE
from keras.optimizers import SGD
from tensorflow.compat.v1 import Session
from tensorflow.compat.v1 import placeholder
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.python.keras.layers import Dense
##v2 tf embaded
from tensorflow.python.keras.models import Sequential
from tensorflow.python.ops.init_ops import GlorotNormal
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2
from tensorflow.python.ops.variable_scope import get_variable
from tensorflow.python.ops.variable_scope import variable_scope
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.training.adam import AdamOptimizer

env = gym.make('CartPole-v1')

np.random.seed(1)
tf.compat.v1.disable_eager_execution()


class ValueNetwork:
    def __init__(self, state_size, learning_rate, num_hidden_layers, num_neurons, name='value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.num_neurons = num_neurons
        self.num_hidden_layers = num_hidden_layers
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.loss_fn = MSE
        self.optimizer = SGD(learning_rate)
        self.model = self._initialize_network()

    def _initialize_network(self):
        model = Sequential()
        model.add(Dense(self.num_neurons, input_dim=self.state_size, name='layer_0'))
        for i in range(1, self.num_hidden_layers):
            model.add(Dense(self.num_neurons,
                            activation='relu', name=f'layer_{i}'))

        model.add(Dense(1, activation='linear'))
        return model

    def train_step(self, x_train, y_train):
        '''
        Performs a gradient descent update of the network.
        Saves performance info for tensorboard.

        params:
            model:  qvalue network to update
            optimizer: optimizer to use
            x_train: states in the minibatch
            y_train: updated y values

        '''
        with tf.GradientTape() as tape:
            predictions = self.model(x_train, training=True)
            loss = self.loss_fn(y_train, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_loss(loss)

    def predict(self, state):
        return self.model.predict(state)


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with variable_scope(name):
            self.state = placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = placeholder(tf.float32, name="total_rewards")
            self.reward_per_episode = placeholder(tf.float32, name="reware_per_episode")
            tf.compat.v1.summary.scalar('rewards', self.reward_per_episode)

            self.W1 = get_variable("W1", [self.state_size, 12], initializer=GlorotNormal(seed=0))
            self.b1 = get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = get_variable("W2", [12, self.action_size], initializer=GlorotNormal(seed=0))
            self.b2 = get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            tf.compat.v1.summary.scalar('loss', self.loss)
            self.optimizer = AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.merged = tf.compat.v1.summary.merge_all()


## Define hyperparameters
state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501

# policy
discount_factor = 0.99
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
ValueFunction = ValueNetwork(state_size, learning_rate_value, num_hidden_layers, num_neurons)

# Start training the agent with REINFORCE algorithm
with Session() as sess:
    sess.run(global_variables_initializer())

    # initiate log files
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = '../logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.compat.v1.summary.FileWriter(train_log_dir, sess.graph)

    solved = False
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0

    for episode in range(max_episodes):
        state = env.reset()
        state = state.reshape([1, state_size])
        episode_transitions = []

        for step in range(max_steps):
            actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])

            if render:
                env.render()

            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1
            episode_transitions.append(
                Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
            episode_rewards[episode] += reward

            if done:
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode,
                                                                                   episode_rewards[episode],
                                                                                   round(average_rewards, 2)))
                if average_rewards > 475:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break
            state = next_state

        if solved:
            break

        # Compute Rt for each time-step t and update the network's weights
        for t, transition in enumerate(episode_transitions):
            total_discounted_return = sum(
                discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))  # Rt
            ValueFunction.train_step(transition.state, np.array(total_discounted_return))

            # base line improvement
            total_discounted_return -= ValueFunction.predict(transition.state)
            feed_dict = {policy.state: transition.state,
                         policy.R_t: total_discounted_return,
                         policy.action: transition.action,
                         policy.reward_per_episode: episode_rewards[episode]}
            _, loss, summary = sess.run([policy.optimizer, policy.loss, policy.merged], feed_dict)

        train_summary_writer.add_summary(summary, episode)
