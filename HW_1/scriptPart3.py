import numpy as np
from recordtype import recordtype
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# from scriptPart2 import CartPoleAgent
from scriptPart2 import CartPoleAgent,gridSearch


class CartPoleAgentPER(CartPoleAgent):

    def __init__(self, memory_size):
        CartPoleAgent.__init__(self, memory_size)
        self.prioritization_epsilon = 0.2
        self.prioritization_alpha = 0.3
        self.Transition = recordtype('Transition',
                                     ['current_state', 'action', 'reward', 'next_state', 'done', "proba", "index"])
        
        
        
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
            model.add(Dense(self.num_neurons, activation='relu', name=f'layer_{i}'))

        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        return model
    def add_experience(self, current_state, action, reward, next_state, done):
        proba = 1
        index = 0
        self.experience_replay.append(self.Transition(current_state, action, reward, next_state, done, proba, index))

        for index, transition in enumerate(self.experience_replay):
            transition.index = index

    def sample_batch(self, minibatch_size):
        transition_probabilities = np.array([transition.proba for transition in self.experience_replay])
        sampling_distribution = transition_probabilities / transition_probabilities.sum()

        minibatch_size = min(minibatch_size, len(sampling_distribution))
        return np.random.choice(self.experience_replay, minibatch_size, p=sampling_distribution, replace=False)

    def _calculate_target_values(self, minibatch, gamma):
        next_state = np.array([transition.next_state[0] for transition in minibatch])
        done = np.array([transition.done for transition in minibatch])
        reward = np.array([transition.reward for transition in minibatch])
        targets = reward + (1 - done) * gamma * np.max(self.target_network.predict(next_state), axis=1)

        current_states = np.array([transition.current_state[0] for transition in minibatch])
        actions = np.array([transition.action for transition in minibatch])
        predictions = self.q_value_network.predict(current_states)
        for (action, target, prediction, transition) in zip(actions, targets, predictions, minibatch):
            p = abs(prediction[action] - target) + self.prioritization_epsilon
            P = p ** self.prioritization_alpha
            self.experience_replay[transition.index].proba = P
            prediction[action] = target
        return predictions


if __name__ == '__main__':
    agent = CartPoleAgentPER(memory_size=1000)
    params = {"num_hidden_layers": [3,5],
              "minibatch_size": [80, 140, 100],
              "gamma": [0.95, 0.9, 0.995],
              "C": [10, 20, 15],
              "epsilon_decay_factor":[0.99, 0.995],
              "epsilon": [0.2, 0.1, 0.4],
              "learning_rate": [0.0001, 0.00001, 0.00005],
              "clipnorm": [True]}
    gridSearch(params, agent, maxN=True)

    # best_params = {'C': 10, 'epsilon': 0.2, 'epsilon_decay_factor': 0.995, 'gamma': 0.95, 'learning_rate': 1e-05,
    #                'minibatch_size': 100, 'num_hidden_layers': 3}
    # agent.train_agent(**best_params, stopEpisode=5000, clipnorm=True)
    # agent.test_agent(100)

    # agent = CartPoleAgentPER(memory_size=10000)
    # agent.train_agent(num_hidden_layers=3)
    # agent.test_agent(100)
