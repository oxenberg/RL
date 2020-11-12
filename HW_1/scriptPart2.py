from collections import namedtuple

import gym
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential, clone_model

import numpy as np


class Dequeu:
    def __init__(self, memorySize):
        self._dequeue = []
        self._N = memorySize

    def append(self, x):
        if len(self._dequeue) == self._N:
            self._dequeue.pop(0)
        self._dequeue.append(x)


Transition = namedtuple('Transition', ['current_state', 'action', 'reward', 'next_state', 'done'])


class CartPoleAgent:
    def __init__(self, memorySize: int):
        self.env = gym.make('CartPole-v1')
        initial_state = self.env.reset()
        self._input_shape = initial_state.shape
        self.numNeurons = 32

        self.experience_replay = Dequeu(memorySize)

    def _initialize_network(self, numHiddenLayers: int):
        model = Sequential()
        model.add(Dense(self.numNeurons, input_dim=self._input_shape), name='layer_0')
        for i in range(1, numHiddenLayers + 1):
            model.add(Dense(self.numNeurons, activation='relu', name=f'layer_{i}'))

        model.add(Dense(self.env.action_space.n, activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def NeuralNetwork(self, state):
        # ToDo: should receive a state or a minibatch
        return self.q_value_network.predict(state)

    def sample_batch(self, minibatch_size):
        '''
        Sample a minibatch randomly from the experience_replay
        :return:
        '''
        return []

    def sample_action(self, state):
        '''
        choose an action with decaying e-greedy method
        :return:
        '''
        pass

    def _has_converged(self) -> bool:
        return np.mean(self.episodes_total_rewards[-100:]) > 475

    def _calculate_target_values(self, minibatch, gamma):
        def get_target(next_state, reward, done):
            if done:
                return reward
            q_values = self.target_network.predict(next_state)
            return reward + gamma * np.max(q_values)

        return [get_target(transition.next_state, transition.reward, transition.done) for transition in minibatch]

    def train_agent(self, numHiddenLayers: int, minibatch_size=10, gamma=0.97, C=10):
        not_converged = True
        self.q_value_network = self._initialize_network(numHiddenLayers)
        self.target_network = clone_model(self.q_value_network)
        self.histories = []
        self.episodes_total_rewards = []
        episodes = 0
        while (not_converged or episodes < 100):
            episodes += 1
            steps = 0
            done = False
            current_state = self.env.reset()
            episode_rewards = []
            while not done:
                action = self.sample_action(current_state)
                next_state, reward, done, info = self.env.step(action)
                steps += 1
                episode_rewards += reward
                self.experience_replay.append(Transition(current_state, action, reward, next_state, done))
                sampled_minibatch = self.sample_batch(minibatch_size)
                y_values = self._calculate_target_values(sampled_minibatch, gamma)
                history = self.q_value_network.fit([transition.current_state for transition in sampled_minibatch],
                                                   y_values,
                                                   epochs=1)
                self.histories.append(history)
                if steps % C == 0:
                    self.target_network = clone_model(self.q_value_network)

            self.episodes_total_rewards.append(sum(episode_rewards))
            not_converged = not self._has_converged()

