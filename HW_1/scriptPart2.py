from collections import namedtuple
from random import shuffle

import gym
from tensorflow.python.keras.layers import Dense,Input
from tensorflow.python.keras.models import Sequential, clone_model
import tensorflow as tf
import numpy as np


class Dequeu:
    def __init__(self, memorySize):
        self._dequeue = []
        self._N = memorySize

    def append(self, x):
        if len(self._dequeue) == self._N:
            self._dequeue.pop(0)
        self._dequeue.append(x)
    def shuffle(self):
        dequeueCopy = self._dequeue.copy()
        shuffle(dequeueCopy)
        return dequeueCopy

Transition = namedtuple('Transition', ['current_state', 'action', 'reward', 'next_state', 'done'])


class CartPoleAgent:
    def __init__(self, memorySize: int):
        self.env = gym.make('CartPole-v1')
        initial_state = self.env.reset()
        self._input_shape = initial_state.shape
        self.numNeurons = 32
        self.num_actions = self.env.action_space.n
        self.actions = [i for i in range(self.env.action_space.n)]

        self.experience_replay = Dequeu(memorySize)

    def _initialize_network(self, numHiddenLayers: int):
        model = Sequential()
        model.add(Dense(self.numNeurons, input_dim=4, name='layer_0'))
        for i in range(1, numHiddenLayers):
            model.add(Dense(self.numNeurons, activation='relu', name=f'layer_{i}'))

        model.add(Dense(self.env.action_space.n, activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def NeuralNetwork(self, state):
        # ToDo: should receive a state or a minibatch
        return self.q_value_network.predict(tf.convert_to_tensor(state))

    def sample_batch(self, minibatch_size):
        '''
        Sample a minibatch randomly from the experience_replay
        :return:
        '''
        return self.experience_replay.shuffle()[:minibatch_size]

    def sample_action(self, state,epsilon):
        '''
        choose an action with decaying e-greedy method
        :return:
        '''
        best_action = np.argmax(self.NeuralNetwork(state))
          
        sampling_distribution = [1 - epsilon if i == best_action else epsilon / (self.num_actions - 1)
                                 for i in range(self.num_actions)]
        return np.random.choice(self.actions, p=sampling_distribution)
        pass

    def _has_converged(self) -> bool:
        return np.mean(self.episodes_total_rewards[-100:]) > 475

    def _calculate_target_values(self,minibatch,gamma):
        def get_target(next_state, reward, done):
            if done:
                return reward
            q_values = self.target_network.predict(np.array([next_state]))
            return reward + gamma * np.max(q_values)

        return [get_target(transition.next_state, transition.reward, transition.done) for transition in minibatch]

    def train_agent(self, numHiddenLayers: int, minibatch_size=10, gamma=0.97, C=10,epsilon = 0.02):
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
            current_state = np.array([self.env.reset()])
            episode_rewards = []
            while not done:
                action = self.sample_action(current_state,epsilon)
                epsilon /= (episodes ** (0.01))
                next_state, reward, done, info = self.env.step(action)
                steps += 1
                episode_rewards .append(reward)
                self.experience_replay.append(Transition(current_state, action, reward, next_state, done))
                sampled_minibatch = self.sample_batch(minibatch_size)
                y_values = self._calculate_target_values(sampled_minibatch, gamma)
                allTransition = []
                for transition in sampled_minibatch:
                    allTransition.append(transition.current_state[0])
                
                history = self.q_value_network.fit(np.array(allTransition),
                                                   np.array(y_values),
                                                   epochs=1)
                self.histories.append(history)
                if steps % C == 0:
                    self.target_network = clone_model(self.q_value_network)

            self.episodes_total_rewards.append(sum(episode_rewards))
            not_converged = not self._has_converged()


if __name__ == '__main__':
    agent = CartPoleAgent(memorySize = 20)
    agent.train_agent(numHiddenLayers = 3)