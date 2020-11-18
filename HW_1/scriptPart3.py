import numpy as np
from recordtype import recordtype

# from scriptPart2 import CartPoleAgent
from HW_1.scriptPart2 import CartPoleAgent


class CartPoleAgentPER(CartPoleAgent):

    def __init__(self, memory_size):
        CartPoleAgent.__init__(self, memory_size)
        self.prioritization_epsilon = 0.01
        self.prioritization_alpha = 0.01
        self.Transition = recordtype('Transition',
                                     ['current_state', 'action', 'reward', 'next_state', 'done', "proba", "index"])

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
    agent = CartPoleAgentPER(memory_size=10000)
    agent.train_agent(num_hidden_layers=3, stopEpisode=200)
    agent.test_agent(100)
