import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import pandas as pd
from random import shuffle
from tqdm import tqdm


SEARCH_HP = True


class FrozenAgent:
    def __init__(self):
        '''
        initiate the env of FrozenLake and the Qtable with zeros

        params:


        -------

        '''
        self.env = gym.make('FrozenLake-v0')
        self.num_actions = self.env.action_space.n
        
        # ToDo: check is_slippery variable in make
        # ToDo: verify on which board size we play
        # ToDo: verify hyperparameter values

        self.actions = [i for i in range(self.env.action_space.n)]
        self.states = [i for i in range(self.env.observation_space.n)]

    def _sampleActionFromQtable(self, state: int, epsilon):
        max_reward = np.max(self.Qtable[state, :])
        best_action = np.random.choice([i for i, q_val in enumerate(self.Qtable[state, :]) if q_val == max_reward])
        sampling_distribution = [1 - epsilon if i == best_action else epsilon / (self.num_actions - 1)
                                 for i in range(self.num_actions)]
        return np.random.choice(self.actions, p=sampling_distribution)

    def train(self, maxEpochs=5000, alpha=0.01, epsilon=0.1, lambd=0.97, maxSteps=100):
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
        self.QtablesSample = {}
        self.rewards = []
        self.stepsPerEpoch = []
        self.Qtable = np.zeros(
                    (self.env.observation_space.n, self.num_actions))
        sampleSteps = [200, 500]
        currentState = self.env.reset()
        overallSteps = 0

        for _ in range(maxEpochs):
            step = 0
            overallReward = 0
            while (True):
                randomAction = self._sampleActionFromQtable(currentState, epsilon)
                newState, reward, done, info = self.env.step(randomAction)
                step += 1
                epsilon *= 1 / step ** (0.01)
                overallReward += reward

                if done or step == maxSteps:
                    self.stepsPerEpoch.append(step)
                    self.rewards.append(overallReward)

                    self.Qtable[currentState, randomAction] += (alpha *
                                                                (reward - self.Qtable[currentState, randomAction]))
                    currentState = self.env.reset()
                    overallSteps += step

                    break

                maxQ = max(self.Qtable[newState])

                target = reward + lambd * maxQ
                self.Qtable[currentState, randomAction] += (alpha *
                                                            (target - self.Qtable[currentState, randomAction]))

                currentState = newState

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
        plt.plot(np.arange(self.maxEpochs), np.cumsum(self.rewards))
        plt.xlabel("Episode number")
        plt.ylabel("Cumulative reward")
        plt.title("Overall Rewards per Episode")

        plt.figure(2)
        try:
            averageWindow = np.mean(
                np.array(self.stepsPerEpoch).reshape(-1, averageOver), axis=1)
            EpisodesSteps = np.arange(0, self.maxEpochs, 100)
            plt.plot(EpisodesSteps, averageWindow)
            plt.xlabel("Episode number")
            plt.ylabel("Steps in the episode")
            plt.title(f"Average steps over {averageOver} episodes")

        except:
            print("cant create graph 2 due of wrong divide number for the window")

        fig, ax = plt.subplots(figsize=(12, 12))
        overallSteps = list(self.QtablesSample.keys())[-1]
        finaleQtable = self.QtablesSample[overallSteps]
        print(finaleQtable)
        im = ax.imshow(finaleQtable)
        states = list(map(str, np.arange(finaleQtable.shape[0])))
        actions = list(map(str, np.arange(finaleQtable.shape[1])))

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(actions)))
        ax.set_yticks(np.arange(len(states)))
        # label them with the respective list entries
        ax.set_xticklabels(actions)
        ax.set_yticklabels(states)
        # Loop over data dimensions and create text annotations.
        for i in range(len(states)):
            for j in range(len(actions)):
                text = ax.text(j, i, round(np.log(finaleQtable[i, j]), 2),
                               ha="center", va="center", color="w")

        ax.set_title("final Q table")
        fig.tight_layout()
        plt.show()


'''
{
 "paramsName" : range(min,max,jump)
 
 }

'''


def gridSearch(parmas, agent, nSearch =10, maxEpochs=5000,maxN = False, aveOver = 10):
    paramsList = list(ParameterGrid(parmas))
    shuffle(paramsList)

    if nSearch > len(paramsList) or maxN:
        nSearch = len(paramsList)

    gridSearchResults = []
    for paramsDict in tqdm(paramsList[:nSearch]):
        aveReward = 0  
        for _ in range(aveOver):
            agent.train(maxEpochs=maxEpochs,
                        alpha=paramsDict["alpha"],
                        epsilon=paramsDict["epsilon"],
                        lambd=paramsDict["lambd"])
            aveReward+= sum(agent.rewards)
        paramsDict['total_reward'] = aveReward/aveOver
        gridSearchResults.append(paramsDict)

    hyperparameterTable = pd.DataFrame(gridSearchResults)
    hyperparameterTable.sort_values("total_reward",inplace = True)
    hyperparameterTable.to_csv("HP.csv")
    print(hyperparameterTable)


if __name__ == '__main__':
    agent = FrozenAgent()
    if SEARCH_HP:
        params = {"alpha": list(np.arange(0.01, 0.05, 0.01)),
                  "epsilon": list(np.arange(0.01, 0.15, 0.01)),
                  "lambd": list(np.arange(0.9, 0.98, 0.01))}
        gridSearch(params, agent, maxN = True)
    else:
        agent.train(alpha = 0.02,
                    epsilon=0.04,
                    lambd=0.96)
        agent.createGraphs()
        print(sum(agent.rewards))
