# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 22:01:28 2020

@author: oxenb
"""


from scriptPart2 import CartPoleAgent
import numpy as np
from recordtype import recordtype


class CartPoleAgentPER(CartPoleAgent):
    
    def __init__(self, memory_size):
        CartPoleAgent.__init__(self, memory_size)
        self.prioritization_epsilon = 0.01
        self.prioritization_alpha = 0.01
        self.Transition = recordtype('Transition', ['current_state', 'action', 'reward', 'next_state', 'done',"proba","index"])

    def add_experience(self,current_state, action, reward, next_state, done):
        proba = 1
        index = 0
        self.experience_replay.append(self.Transition(current_state, action, reward, next_state, done,proba,index))
        
        for index,transition in enumerate(self.experience_replay):
            transition.index = index
            
    def sample_batch(self, minibatch_size):
        sampling_distribution = [transition.proba for transition in self.experience_replay]
        normalized_p = sum([transition.proba for transition in self.experience_replay])
        sampling_distribution = [sample/normalized_p for sample in sampling_distribution]
        minibatch_size = min(minibatch_size,len(sampling_distribution))
        
        return np.random.choice( self.experience_replay,minibatch_size, p=sampling_distribution,replace=False)
    def _calculate_target_values(self, minibatch, gamma):
        next_state = np.array([transition.next_state[0] for transition in minibatch])

        done = np.array([transition.done for transition in minibatch])
        reward = np.array([transition.reward for transition in minibatch])
        targets = reward + (1 - done) * gamma * np.max(self.target_network.predict(next_state), axis=1)
        
        current_states = np.array([transition.current_state[0] for transition in minibatch])
        actions = np.array([transition.action for transition in minibatch])
        predictions = self.q_value_network.predict(current_states)
        predictions = self.q_value_network.predict(current_states)
        for (action,target,prediction,transition) in zip(actions,targets,predictions,minibatch):
            p = abs(prediction[action] - target) + self.prioritization_epsilon
            P = p**self.prioritization_alpha
            self.experience_replay[transition.index].proba = P
            prediction[action] = target
        return predictions
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
    agent = CartPoleAgentPER(memory_size=10000)
    agent.train_agent(num_hidden_layers=3,stopEpisode=200)
    agent.test_agent(100)