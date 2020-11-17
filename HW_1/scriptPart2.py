from collections import namedtuple, deque, defaultdict
from random import shuffle, sample

import gym
import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams.api import HParam, RealInterval, Discrete, hparams_config, Metric
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential, clone_model
from tensorflow.keras.optimizers import Adam

import datetime
import keras

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)


MIN_EPSILON = 0.001

class CartPoleAgent:
    def __init__(self, memory_size):
        self.env = gym.make('CartPole-v1')
        initial_state = self.env.reset()
        self._input_shape = initial_state.shape
        self.num_neurons = 12
        self.num_actions = self.env.action_space.n
        self.actions = [i for i in range(self.env.action_space.n)]
        self.convergedTH = 475
        self.Transition = namedtuple('Transition', ['current_state', 'action', 'reward', 'next_state', 'done'])

        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

        self.loss_fn = keras.losses.MSE

        
        self.experience_replay = deque(maxlen=memory_size)

    def _initialize_network(self, num_hidden_layers: int, learning_rate: float):
        model = Sequential()
        model.add(Dense(self.num_neurons, input_dim=4, name='layer_0'))
        for i in range(1, num_hidden_layers):
            model.add(Dense(self.num_neurons, activation='relu', name=f'layer_{i}'))

        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        return model

    def NeuralNetwork(self, state):
        # ToDo: should receive a state or a minibatch
        return self.q_value_network.predict(state)

    def sample_batch(self, minibatch_size):
        '''
        Sample a minibatch randomly from the experience_replay
        :return:
        '''
        return sample(self.experience_replay, min(minibatch_size, len(self.experience_replay)))

    def sample_action(self, state, epsilon):
        '''
        choose an action with decaying e-greedy method
        :return:
        '''
        best_action = np.argmax(self.NeuralNetwork(state))
        sampling_distribution = [1 - epsilon if i == best_action else epsilon / (self.num_actions - 1)
                                 for i in range(self.num_actions)]
        return np.random.choice(self.actions, p=sampling_distribution)

    def _has_converged(self) -> bool:
        return np.mean(self.episodes_total_rewards[-100:]) >= self.convergedTH

    def _calculate_target_values(self, minibatch, gamma):
        next_state = np.array([transition.next_state[0] for transition in minibatch])

        done = np.array([transition.done for transition in minibatch])
        reward = np.array([transition.reward for transition in minibatch])
        targets = reward + (1 - done) * gamma * np.max(self.target_network.predict(next_state), axis=1)
        
        current_states = np.array([transition.current_state[0] for transition in minibatch])
        actions = np.array([transition.action for transition in minibatch])
        predictions = self.q_value_network.predict(current_states)
        for action,target,prediction in zip(actions,targets,predictions):
            prediction[action] = target
       
        return predictions
    def train_step(self,model, optimizer, x_train, y_train):
        with tf.GradientTape() as tape:
            predictions = model(x_train, training=True)
            loss = self.loss_fn(y_train, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        self.train_loss(loss)
        
    def _update_target_network(self):
        for l_tg,l_sr in zip(self.target_network.layers,self.q_value_network.layers):
            wk0=l_sr.get_weights()
            l_tg.set_weights(wk0)
        
        
    def add_experience(self,current_state, action, reward, next_state, done):
        self.experience_replay.append(self.Transition(current_state, action, reward, next_state, done))

    def train_agent(self,
                    num_hidden_layers: int,
                    minibatch_size=20,
                    gamma=0.95,
                    C=10,
                    epsilon=0.8,
                    learning_rate=0.0005,
                    stopEpisode = None):
        not_converged = True
        self.optimizer = keras.optimizers.SGD(learning_rate)

        self.q_value_network = self._initialize_network(num_hidden_layers, learning_rate)
        self.target_network = self._initialize_network(num_hidden_layers, learning_rate)
        self.episodes_total_rewards = []
        episodes = 0
        steps = 0
        while ((not_converged or episodes < 100) and episodes != stopEpisode):
            episodes += 1
            done = False
            current_state = np.array([self.env.reset()])
            episode_rewards = 0
            while not done:
                action = self.sample_action(current_state, epsilon)
                epsilon = max(epsilon*0.99,MIN_EPSILON)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.array([next_state])
                steps += 1
                episode_rewards += reward
                self.add_experience(current_state, action, reward, next_state, done)
                current_state = next_state

                sampled_minibatch = self.sample_batch(minibatch_size)
                y_values = self._calculate_target_values(sampled_minibatch, gamma)
                # print(y_values)
                all_transitions = np.array([transition.current_state[0] for transition in sampled_minibatch])
                
            
                self.train_step(self.q_value_network,self.optimizer,all_transitions,y_values)
                
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=steps)
                

                if steps % C == 0:
                    self._update_target_network()

            self.episodes_total_rewards.append(episode_rewards)
            with train_summary_writer.as_default():
                tf.summary.scalar('rewards', data=episode_rewards, step=episodes)
                tf.summary.scalar('epsilon', data=epsilon, step=episodes)


            if episodes % 10 == 0:
                print(episodes, np.mean(self.episodes_total_rewards[-100:]))
            not_converged = not self._has_converged()
    def test_agent(self,episodes,render = False):
        overallReward = 0
        for i in range(episodes):
            done = False
            aggrigateReward = 0
            current_state = np.array([self.env.reset()])
            while not done:
                action = self.sample_action(current_state, 0)
                next_state, reward, done, _ = self.env.step(action)
                aggrigateReward+=reward
                if render:
                    self.env.render()
            overallReward+=aggrigateReward
        aveReward = overallReward/episodes
        print(f"average reward is {aveReward}")
                    



if __name__ == '__main__':
    # for memory_size in range(10, 100, 10):
        # HP_HIDDEN_LAYERS = HParam('num_hidden_layers', Discrete([3, 5]))
        # HP_MINIBATCH_SIZE = HParam('minibatch_size', RealInterval(0.1 * memory_size, 0.9 * memory_size))
        # HP_GAMMA = HParam('gamma', RealInterval(0.95, 1.0))
        # HP_C = HParam('C', Discrete(list(range(10, 100, 10))))
        # HP_EPSILON = HParam('epsilon', RealInterval(0.001, 0.05))
        # HP_LEARNING_RATE = HParam('learning_rate', RealInterval(0.0001, 0.01))
        #
        # log_dir = rf'.\logs\memory_size_{memory_size}'
        # with tf.summary.create_file_writer(log_dir).as_default():
        #     hparams_config(
        #         hparams=[HP_HIDDEN_LAYERS, HP_MINIBATCH_SIZE, HP_GAMMA, HP_C, HP_EPSILON, HP_LEARNING_RATE],
        #         metrics=[Metric('MeanSquaredError', display_name='MSE')],
        #     )
    agent = CartPoleAgent(memory_size=10000)
    agent.train_agent(num_hidden_layers=3,stopEpisode=200)
    agent.test_agent(100)