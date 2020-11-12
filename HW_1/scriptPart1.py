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
        self.Qtable = np.zeros(
            (self.env.observation_space.n, self.num_actions))
        # ToDo: check is_slippery variable in make
        # ToDo: verify on which board size we play
        # ToDo: verify hyperparameter values

        self.actions = [i for i in range(self.env.action_space.n)]
        self.states = [i for i in range(self.env.observation_space.n)]


    def _sampleActionFromQtable(self, state: int, epsilon):
        best_action = np.argmax(self.Qtable[state, :])
        sampling_distribution = [1 - epsilon if i == best_action else epsilon / (self.num_actions - 1)
                                 for i in range(self.num_actions)]
        return np.random.choice(self.actions, p=sampling_distribution)

    def train(self, maxEpochs=10, alpha=0.01,epsilon = 0.98, lambd=0.01, maxSteps=100):
        '''
        train the agent on the env with the Q-learning algo
        
        params:

        maxEpochs (float) -
        alpha (float) -
        lambd (float) -
        epsilon(float) - for epsilon greedy sampling algorithm

        Returns
        -------
        None.

        '''
        self.maxEpochs = maxEpochs
        self.rewards = []
        self.stepsPerEpoch = []
        self.QtablesSample = {}

        sampleSteps = [200, 500]
        currentState = self.env.reset()
        overallSteps = 0
        self.rewards = []
        self.stepsPerEpoch = []

        currentState = self.env.reset()
        for _ in range(maxEpochs):
            step = 0
            overallReward = 0
            while (True):
                randomAction = self._sampleActionFromQtable(
                    currentState, epsilon)
                newState, reward, done, info = self.env.step(randomAction)
                overallReward += reward

                
                if done or step == maxSteps:
                    self.stepsPerEpoch.append(step)
                    self.rewards.append(overallReward)

                    self.Qtable[currentState, randomAction] = alpha * \
                        (reward - self.Qtable[currentState, randomAction])
                    currentState = self.env.reset()
                    overallSteps += step

                    break

                maxQ = max(self.Qtable[newState])

                target = reward + lambd * maxQ
                self.Qtable[currentState, randomAction] += alpha * \
                    (target - self.Qtable[currentState, randomAction])

                currentState = newState
                step += 1
                # self.env.render()

            if overallSteps in sampleSteps:
                self.QtablesSample[overallSteps] = self.Qtable

        self.QtablesSample[overallSteps] = self.Qtable
        self.env.close()

    def createGraphs(self, averageOver=100):
        '''
        1.Plot of the reward per episode.
        2.Plot of the average number of steps to the goal over last 100 episodes

        Returns
        -------
        None.

        '''
        plt.figure(1)
        plt.plot(np.arange(self.maxEpochs), self.rewards)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.title("reward per episode")

        plt.figure(2)
        try:
            averageWindow = np.mean(
                np.array(self.stepsPerEpoch).reshape(-1, averageOver), axis=1)
            EpisodesSteps = np.arange(0, self.maxEpochs, 100)
            plt.plot(EpisodesSteps, averageWindow)
            plt.xlabel("episode")
            plt.ylabel("[steps]")
            plt.title(f"average steps over {averageOver} episode")

        except:
            print("cant create graph 2 due of wrong divide number for the window")

        fig, ax = plt.subplots(figsize=(12, 12))
        overallSteps = list(self.QtablesSample.keys())[-1]
        finaleQtable = self.QtablesSample[overallSteps]
        print(finaleQtable)
        im = ax.imshow(finaleQtable)
        states = list(map(str,np.arange(finaleQtable.shape[0])))
        actions = list(map(str,np.arange(finaleQtable.shape[1])))

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(actions)))
        ax.set_yticks(np.arange(len(states)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(actions)
        ax.set_yticklabels(states)
        # Loop over data dimensions and create text annotations.
        for i in range(len(states)):
            for j in range(len(actions)):
                text = ax.text(j, i, round(np.log(finaleQtable[i, j]),2),
                               ha="center", va="center", color="w")

        ax.set_title("final Q table")
        fig.tight_layout()
        plt.show()


# env = gym.make('FrozenLake-v0')
# env.reset()
# env_parms = 0
# for _ in range(100):
#     env.render()
#     observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
# env.close()


if __name__ == '__main__':
    agent = FrozenAgent()
    agent.train(maxEpochs = 5000)
    agent.createGraphs()
