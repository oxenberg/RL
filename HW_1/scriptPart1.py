import gym
import numpy as np
import matplotlib.pyplot as plt


class FrozenAgent:
    def __init__(self):
        '''
        initiate the env of FrozenLake and the Qtable with zeros

        params:


        -------

        '''
        self.env = gym.make('FrozenLake-v0')

        self.num_actions = self.env.action_space.n
        self.Qtable = np.zeros((self.env.observation_space.n, self.num_actions))
        # ToDo: check is_slippery variable in make
        # ToDo: verify on which board size we play

        self.actions = [i for i in range(self.env.action_space.n)]
        self.states = [i for i in range(self.env.observation_space.n)]

    def _sampleActionFromQtable(self, state: int, epsilon=0.1):
        best_action = np.argmax(self.Qtable[state, :])
        sampling_distribution = [1 - epsilon if i == best_action else epsilon / (self.num_actions - 1)
                                 for i in range(self.num_actions)]
        return np.random.choice(self.actions, p=sampling_distribution)

    def train(self):
        self.env = gym.make('FrozenLake-v0')
        self.Qtable = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def train(self, maxEpochs=10, alpha=0.01, lambd=0.97):
        '''
        params:

        maxEpochs (float) -
        alpha (float) -
        lambd (float) -

        Returns
        -------
        None.

        '''
        # ToDo: verify hyperparameter values
        self.maxEpochs = 10
        self.rewards = []
        self.stepsPerEpoch = []

        currentState = self.env.reset()

        for _ in range(maxEpochs):
            step = 0
            while (True):
                randomAction = self.env.action_space.sample()
                newState, reward, done, info = self.env.step(randomAction)
                maxQ = max(self.Qtable[newState])
                target = reward + lambd * maxQ
                self.Qtable[currentState, randomAction] += alpha * (target - self.Qtable[currentState, randomAction])

                currentState = newState
                step += 1
                self.env.render()
                if done:
                    self.rewards.append(reward)
                    currentState = self.env.reset()
                    break
        self.env.close()

    def createGraphs(self):
        '''
        1.Plot of the reward per episode.
        2.Plot of the average number of steps to the goal over last 100 episodes

        Returns
        -------
        None.

        '''
        plt.figure(1)
        plt.plot(np.arange(self.maxEpochs), self.rewards)

# env = gym.make('FrozenLake-v0')
# env.reset()
# env_parms = 0
# for _ in range(100):
#     env.render()
#     observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
# env.close()
