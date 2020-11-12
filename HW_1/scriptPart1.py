# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 20:40:20 2020

@author: oxenb
"""
import gym
import numpy as np
import matplotlib.pyplot as plt

class FrozenAgent():
    def __init__(self):
        '''
        initiate the env of FrozenLake and the Qtable with zeros
        
        parans:
 

        -------

        '''
        self.env = gym.make('FrozenLake-v0')
        self.Qtable = np.zeros((self.env.observation_space.n,self.env.action_space.n))

        
    def train(self,maxEpochs = 10,alpha = 0.01,lambd = 0.97,maxSteps = 100):
        '''
        params:
            
        maxEpochs (float) -
        alpha (float) - 
        lambd (float) -
            
        Returns
        -------
        None.

        '''
        self.maxEpochs = 800
        self.rewards = [] 
        self.stepsPerEpoch = []
        self.QtablesSample = {}
        
        sampleSteps = [200,500]
        currentState = self.env.reset()
        overallSteps = 0
        for epoch in range(maxEpochs):
            step = 0
            while(True):
                
                randomAction = self.env.action_space.sample()
                newState, reward, done, info = self.env.step(randomAction)
                
                if done or step ==maxSteps:
                    self.stepsPerEpoch.append(step)
                    self.rewards.append(reward)
                    
                    self.Qtable[currentState,randomAction] = alpha*(reward - self.Qtable[currentState,randomAction])
                    currentState = self.env.reset()
                    overallSteps+=step
                    
                    break
                
                maxQ = max(self.Qtable[newState])
                
                target = reward + lambd * maxQ
                self.Qtable[currentState,randomAction] += alpha*(target - self.Qtable[currentState,randomAction])
                
                currentState = newState
                step+=1
                self.env.render()
            
            if overallSteps in sampleSteps:
                self.QtablesSample[overallSteps] = self.Qtable

        self.QtablesSample[overallSteps] = self.Qtable
        self.env.close()

        
    def createGraphs(self,averageOver = 100):
        '''
        1.Plot of the reward per episode.
        2.Plot of the average number of steps to the goal over last 100 episodes

        Returns
        -------
        None.

        '''
        plt.figure(1)
        plt.plot(np.arange(self.maxEpochs),self.rewards)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.title("reward per episode")

        plt.figure(2)
        try:
            averageWindow = np.mean(np.array(self.stepsPerEpoch).reshape(-1,averageOver ), axis=1)
            EpisodesSteps = np.arange(0,len(self.maxEpochs),100)
            plt.plot(EpisodesSteps,averageWindow)
            plt.xlabel("episode")
            plt.ylabel("[steps]")
            plt.title(f"average steps over {averageOver} episode")

        
        except:
            print("cant create graph 2 due of wrong divide number for the window")


        fig, ax = plt.subplots(3)
        overallSteps = self.QtablesSample.keys()[-1]
        finaleQtable = self.QtablesSample[overallSteps]
        im = ax.imshow(finaleQtable)
        
               
        # Loop over data dimensions and create text annotations.
        for i in range(len(finaleQtable)):
            for j in range(len(finaleQtable)):
                text = ax.text(j, i, finaleQtable[i, j],
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


















