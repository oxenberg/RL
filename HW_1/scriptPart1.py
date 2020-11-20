from random import shuffle

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

SEARCH_HP = False


class FrozenAgent:
    def __init__(self):
        '''
        An agent for the FrozenLake OpenGym environment.
        Learns to play using the Q-learning algorithm.
        Initiates the FrozenLake environment.
        '''
        self.env = gym.make('FrozenLake-v0')
        self.num_actions = self.env.action_space.n
        self.actions = [i for i in range(self.env.action_space.n)]
        self.states = [i for i in range(self.env.observation_space.n)]

    def _sampleActionFromQtable(self, state: int, epsilon: float):
        '''
        Samples an action based on the Qtable using an epsilon-greedy approach.

        params:
            state (int): the current state for which we need to act on
            epsilon (float): current epsilon value for the epsilon-greedy sampling

        return:
            action (int): the action that was sampled

        '''
        max_reward = np.max(self.Qtable[state, :])
        best_action = np.random.choice(
            [i for i, q_val in enumerate(self.Qtable[state, :]) if q_val == max_reward])
        sampling_distribution = [1 - epsilon if i == best_action else epsilon / (self.num_actions - 1)
                                 for i in range(self.num_actions)]
        return np.random.choice(self.actions, p=sampling_distribution)

    def train(self, maxEpochs=5000, alpha=0.01, epsilon=0.1, gamma=0.97, maxSteps=100):
        '''
        Trains the agent on the FrozenLake environment with the Q-learning algorithm.
        At each step, samples an action using decaying epsilon-greedy approach.

        params:
            maxEpochs (int): number of epochs to train on
            alpha (float): learning rate for the update steps
            gamma (float): decay factor
            epsilon(float): initial epsilon value for epsilon-greedy sampling of the action
            masSteps (int): maximum steps per episode

        return:
            None

        '''
        self.maxEpochs = maxEpochs
        self.QtablesSample = {}
        self.rewards = []
        self.stepsPerEpoch = []
        self.Qtable = np.zeros((self.env.observation_space.n, self.num_actions))
        sampleSteps = [200, 500]
        currentState = self.env.reset()
        overallSteps = 0

        for _ in range(maxEpochs):
            step = 0
            overallReward = 0
            while (True):
                randomAction = self._sampleActionFromQtable(
                    currentState, epsilon)
                newState, reward, done, info = self.env.step(randomAction)
                step += 1
                overallSteps += 1
                epsilon *= 1 / step ** 0.01
                overallReward += reward

                if done or step == maxSteps:
                    epoch_steps = step if newState == 15 else maxSteps
                    self.stepsPerEpoch.append(epoch_steps)
                    self.rewards.append(overallReward)

                    self.Qtable[currentState, randomAction] += (alpha *
                                                                (reward - self.Qtable[currentState, randomAction]))
                    currentState = self.env.reset()
                    if overallSteps in sampleSteps:
                        self.QtablesSample[overallSteps] = self.Qtable
                    break

                maxQ = max(self.Qtable[newState])
                target = reward + gamma * maxQ
                self.Qtable[currentState, randomAction] += (alpha *
                                                            (target - self.Qtable[currentState, randomAction]))

                currentState = newState
                if overallSteps in sampleSteps:
                    self.QtablesSample[overallSteps] = self.Qtable

        self.QtablesSample[overallSteps] = self.Qtable
        self.env.close()

    def createGraphs(self, averageOver=100):
        '''
        Creates graphs of the agent's learning process. Can use after the agent has been trained.
        Will show the following plots:
            1. Plot of the cumulative rewards per episode.
            2. Plot of the average number of steps to the goal over last 100 episodes
            3. Colormaps of the Q-value table after 500 steps, 2000 steps and the final table
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

        for steps, q_table in self.QtablesSample.items():
            fig, ax = plt.subplots(figsize=(12, 12))
            _ = ax.imshow(q_table)
            states = list(map(str, np.arange(q_table.shape[0])))
            actions = list(map(str, np.arange(q_table.shape[1])))

            # We want to show all ticks...
            ax.set_xticks(np.arange(len(actions)))
            ax.set_yticks(np.arange(len(states)))
            # label them with the respective list entries
            ax.set_xticklabels(actions)
            ax.set_yticklabels(states)
            # Loop over data dimensions and create text annotations.
            for i in range(len(states)):
                for j in range(len(actions)):
                    _ = ax.text(j, i, round(np.log(q_table[i, j]), 2),
                                ha="center", va="center", color="w")

            title = f"Qtable after {steps} steps" if steps in (200, 500) else "Final Qtable"
            ax.set_title(title)
            fig.tight_layout()
        plt.show()


def gridSearch(params, agent, nSearch=10, maxEpochs=5000, maxN=False, aveOver=10):
    paramsList = list(ParameterGrid(params))
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
                        gamma=paramsDict["gamma"])
            aveReward += sum(agent.rewards)
        paramsDict['total_reward'] = aveReward / aveOver
        gridSearchResults.append(paramsDict)

    hyperparameterTable = pd.DataFrame(gridSearchResults)
    hyperparameterTable.sort_values("total_reward", inplace=True)
    hyperparameterTable.to_csv("HP.csv")
    print(hyperparameterTable)


if __name__ == '__main__':
    agent = FrozenAgent()
    if SEARCH_HP:
        params = {"alpha": list(np.arange(0.01, 0.05, 0.01)),
                  "epsilon": [0.8, 0.5, 0.2],
                  "gamma": list(np.arange(0.9, 0.98, 0.01))}
        gridSearch(params, agent, maxN=True)
    else:
        agent.train(alpha=0.01,
                    epsilon=0.8,
                    gamma=0.9)
        agent.createGraphs()
        print(sum(agent.rewards))
