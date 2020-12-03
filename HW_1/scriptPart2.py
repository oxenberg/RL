import datetime
from collections import deque
from collections import namedtuple
from random import sample
from random import shuffle

import gym
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from keras.optimizers import SGD
from tqdm import tqdm

SEARCH_HP = False


class CartPoleAgent:
    def __init__(self, memory_size):
        '''
        An agent for the Cartpole-v1 OpenGym environment.
        Learns to play using the DQL algorithm with experience replay.
        Initiates the Cartpole environment and agent parameters.

        params:
            memory_size (int): size for the experience replay deque

        '''
        self.env = gym.make('CartPole-v1')
        initial_state = self.env.reset()
        self._input_shape = initial_state.shape
        self.num_neurons = 26
        self.num_actions = self.env.action_space.n
        self.actions = [i for i in range(self.env.action_space.n)]
        self.convergedTH = 475
        self.Transition = namedtuple(
            'Transition', ['current_state', 'action', 'reward', 'next_state', 'done'])

        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.loss_fn = keras.losses.MSE

        self.experience_replay = deque(maxlen=memory_size)

    def _initialize_network(self, num_hidden_layers: int, learning_rate: float):
        '''
        Creates a neural network for the q-value function and the target function.
        Each layer will have 24 neurons. The hidden layers use a RELU activation function while the
        output layer uses a Linear activation function.
        The Adam optimization algorithm is used for the back propagation.

        params:
            num_hidden_layers (int): number of hidden layers in the model
            learning_rate (float): learning rate for the Adam optimizer

        return:
            model: a compiled model

        '''
        model = Sequential()
        model.add(Dense(self.num_neurons, input_dim=4, name='layer_0'))
        for i in range(1, num_hidden_layers):
            model.add(Dense(self.num_neurons,
                            activation='relu', name=f'layer_{i}'))

        model.add(Dense(self.env.action_space.n, activation='linear'))
        return model

    def NeuralNetwork(self, state):
        '''
        Gets the q-value estimation based on the current q-network.

        params:
            state (ndarray): a state or a batch of states

        return:
            ndarray: current q-value estimations

        '''
        return self.q_value_network.predict(state)

    def sample_batch(self, minibatch_size):
        '''
        Samples a minibatch randomly from the experience_replay

        params:
            minibatch_size (int): number of transitions to sample

        return:
            list: random list of transitions
        '''
        return sample(self.experience_replay, min(minibatch_size, len(self.experience_replay)))

    def sample_action(self, state, epsilon):
        '''
        Chooses an action with decaying e-greedy method.

        params:
            state (ndarray): representation of the current state
            epsilon (float): current epsilon for the e-greedy method.

        return:
            int: the action to take
        '''
        best_action = np.argmax(self.NeuralNetwork(state))
        sampling_distribution = [1 - epsilon if i == best_action else epsilon / (self.num_actions - 1)
                                 for i in range(self.num_actions)]
        return np.random.choice(self.actions, p=sampling_distribution)

    def _has_converged(self) -> bool:
        '''
        Checks if the average rewards in the last 100 episodes is larger than 475

        return:
            bool: True if the network achieved the goal
        '''
        return np.mean(self.episodes_total_rewards[-100:]) >= self.convergedTH

    def _calculate_target_values(self, minibatch, gamma):
        '''
        Calculates the updated y values for the gradient descent step.

        params:
            minibatch (list[transition]): sampled minibatch for the q-value network update
            gamma (float): decay factor

        return:
            ndarray: updated y values for the gradient descent
        '''
        next_state = np.array([transition.next_state[0]
                               for transition in minibatch])
        done = np.array([transition.done for transition in minibatch])
        reward = np.array([transition.reward for transition in minibatch])
        targets = reward + (1 - done) * gamma * \
            np.max(self.target_network.predict(next_state), axis=1)

        current_states = np.array([transition.current_state[0]
                                   for transition in minibatch])
        actions = np.array([transition.action for transition in minibatch])
        predictions = self.q_value_network.predict(current_states)
        for action, target, prediction in zip(actions, targets, predictions):
            prediction[action] = target

        return predictions

    def train_step(self, model, optimizer, x_train, y_train):
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
            predictions = model(x_train, training=True)
            loss = self.loss_fn(y_train, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        self.train_loss(loss)

    def _update_target_network(self):
        '''
        Updates the target network's weights to the weights of the q-value network.
        '''
        for l_tg, l_sr in zip(self.target_network.layers, self.q_value_network.layers):
            wk0 = l_sr.get_weights()
            l_tg.set_weights(wk0)

    def add_experience(self, current_state, action, reward, next_state, done):
        '''
        Adds a transition to the experience replay deque.

        params:
            current_state (ndarray): state where the transition originated from
            action (int): action taken
            reward (int): reward observed
            next_state (ndarray): state arrived to after taking the action
            done (bool): is the episode done
        '''
        self.experience_replay.append(self.Transition(
            current_state, action, reward, next_state, done))

    def init_log_files(self):
        '''
        Initiates the log files for tensorboard
        '''
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = 'logs/gradient_tape/' + self.current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(
            self.train_log_dir)

    def train_agent(self,
                    num_hidden_layers: int,
                    minibatch_size=20,
                    gamma=0.95,
                    C=10,
                    epsilon=0.8,
                    learning_rate=0.0005,
                    stopEpisode=None,
                    epsilon_decay_factor=0.99,
                    clipnorm=None):
        '''
        Trains the agent to play on the cartpole environment using experience replay.
        Samples an action based on a decaying epsilon-greedy method.

        params:
            num_hidden_layers (int): number of hidden layers for the qvalue and target network.
            minibatch_size (int): minibatch size for sampling from the experience replay
            gamma (float): decay factor
            C (int): number of iterations for target network update
            epsilon (float): initial epsilon value for epsilon-greedy sampling of the action
            learning_rate (float): learning rate for the network.
            stopEpisode (int): maximum number of episodes
            epsilon_decay_factor (float): factor for decaying-epsilon sampling
        '''
        not_converged = True
        self.init_log_files()

        if clipnorm:
            self.optimizer = SGD(learning_rate, clipnorm=1.0)
        else:
            self.optimizer = SGD(learning_rate)

        self.q_value_network = self._initialize_network(
            num_hidden_layers, learning_rate)
        self.target_network = self._initialize_network(
            num_hidden_layers, learning_rate)
        self.episodes_total_rewards = []
        episodes = 0
        steps = 0
        while (not_converged or episodes < 100) and episodes != stopEpisode:
            episodes += 1
            done = False
            current_state = np.array([self.env.reset()])
            episode_rewards = 0
            while not done:
                action = self.sample_action(current_state, epsilon)
                epsilon *= epsilon_decay_factor
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.array([next_state])
                steps += 1
                episode_rewards += reward
                self.add_experience(current_state, action,
                                    reward, next_state, done)
                current_state = next_state

                sampled_minibatch = self.sample_batch(minibatch_size)
                y_values = self._calculate_target_values(
                    sampled_minibatch, gamma)
                all_transitions = np.array(
                    [transition.current_state[0] for transition in sampled_minibatch])

                self.train_step(self.q_value_network,
                                self.optimizer, all_transitions, y_values)

                with self.train_summary_writer.as_default():
                    tf.summary.scalar(
                        'loss', self.train_loss.result(), step=steps)

                if steps % C == 0:
                    self._update_target_network()

            self.episodes_total_rewards.append(episode_rewards)
            with self.train_summary_writer.as_default():
                tf.summary.scalar(
                    'rewards', data=episode_rewards, step=episodes)
                tf.summary.scalar('epsilon', data=epsilon, step=episodes)

            not_converged = not self._has_converged()

    def test_agent(self, episodes, render=False):
        '''
        Test the trained agent. Can see how well he plays as a result of the training process.

        params:
            episodes (int): number of episodes to run
            render (bool): render the environment
        '''
        overallReward = 0
        for _ in range(episodes):
            done = False
            aggrigateReward = 0
            current_state = np.array([self.env.reset()])
            while not done:
                action = self.sample_action(current_state, 0)
                _, reward, done, _ = self.env.step(action)
                aggrigateReward += reward
                if render:
                    self.env.render()
            overallReward += aggrigateReward
        aveReward = overallReward / episodes
        print(f"average reward is {aveReward}")


def gridSearch(parmas, agent, nSearch=10, maxEpochs=5000, maxN=False, aveOver=10):
    paramsList = list(ParameterGrid(parmas))
    shuffle(paramsList)

    if nSearch > len(paramsList) or maxN:
        nSearch = len(paramsList)

    gridSearchResults = []
    for paramsDict in tqdm(paramsList[:nSearch]):
        try:
            agent.train_agent(stopEpisode=1000, **paramsDict)
            paramsDict['average_reward_last_100_episodes'] = np.mean(
                agent.episodes_total_rewards[-100:])
            print(paramsDict)
            gridSearchResults.append(paramsDict)
        except Exception as e:
            print(e)
            continue

    hyperparameterTable = pd.DataFrame(gridSearchResults)
    hyperparameterTable.sort_values("total_reward", inplace=True)
    hyperparameterTable.to_csv("Part2-HP.csv")
    print(hyperparameterTable)


if __name__ == '__main__':
    agent = CartPoleAgent(memory_size=2000)
    if SEARCH_HP:
        params = {"num_hidden_layers": [3, 5],
                  "minibatch_size": [20, 50, 70, 100],
                  "gamma": [0.95, 0.9, 0.99],
                  "C": [10, 20, 15],
                  "epsilon_decay_factor": [0.99, 0.995],
                  "epsilon": [0.8, 0.5, 0.2],
                  "learning_rate": [0.0001, 0.001, 0.00001]}
        gridSearch(params, agent, maxN=True)

    best_params = {'C': 10, 'epsilon': 0.2, 'epsilon_decay_factor': 0.995, 'gamma': 0.95, 'learning_rate': 1e-05,
                   'minibatch_size': 100, 'num_hidden_layers': 3}
    agent.train_agent(**best_params, stopEpisode=5000)
    agent.test_agent(100)
