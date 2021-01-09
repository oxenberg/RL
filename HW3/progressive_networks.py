from datetime import datetime
from random import shuffle

import gym
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm

from tensorflow import concat
from tensorflow.compat.v1 import Session
from tensorflow.compat.v1 import placeholder
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.python.ops.distributions.normal import Normal
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2
from tensorflow.python.ops.variable_scope import variable_scope
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.saver import Saver

from actor_critic import ACTION_SIZE
from actor_critic import ENV_TO_ACTION_SIZE
from actor_critic import ENV_TO_REWARD_THRESHOLD
from actor_critic import ENV_TO_STATE_SIZE
from actor_critic import OpenGymEnvs
from actor_critic import PolicyNetwork
from actor_critic import STATE_SIZE
from actor_critic import ValueNetwork
from actor_critic import scale_state


from train_mountain_car import getBestParamsMountain
from train_acrobot import getBestParamsAcro
from train_cartpole import getBestParamsCart

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

MAX_EPISODES = 700


class ProgressivePolicyNetwork:
    def __init__(self,
                 learning_rate,
                 policy_source1: PolicyNetwork,
                 policy_source2: PolicyNetwork,
                 policy_out: PolicyNetwork,
                 mountain_car=None,
                 name='progressive_policy_network'):
        self.learning_rate = learning_rate
        self.name = name
        self.init = tf.initializers.GlorotUniform()

        with variable_scope(self.name):
            self.action = placeholder(tf.float32, [ACTION_SIZE], name="action")
            self.R_t = placeholder(tf.float32, name="total_rewards")
            self.reward_per_episode = placeholder(tf.float32, name="reware_per_episode")
            tf.compat.v1.summary.scalar('rewards', self.reward_per_episode)

          
            Z2_1 = tf.add(tf.matmul(policy_source1.A1, policy_source1.W2), policy_source1.b2)
            Z2_2 = tf.add(tf.matmul(policy_source2.A1, policy_source2.W2), policy_source2.b2)
            Z2_3 = tf.add(tf.matmul(policy_out.A1, policy_out.W2), policy_out.b2)
            
            if mountain_car:
                self.output_mu = tf.math.add_n([Z2_1, Z2_2, Z2_3])
                self.output_var = tf.math.add_n([Z2_1, Z2_2, Z2_3])

                self.output_mu = tf.squeeze(self.output_mu)
                self.output_var = tf.squeeze(self.output_var)
                self.output_var = tf.nn.softplus(self.output_var) + 1e-5
                self.normal_dist = Normal(self.output_mu, self.output_var)
                self.sampled_action = tf.squeeze(
                    self.normal_dist._sample_n(1), axis=0)
                self.actions_distribution = tf.clip_by_value(self.sampled_action, -1, 1)
                # Loss and train op

                self.loss = - tf.math.log(self.normal_dist.prob(
                    tf.squeeze(self.action)) + 1e-5) * self.R_t
            else:
                self.output = tf.math.add_n([Z2_1, Z2_2, Z2_3])
                # Softmax probability distribution over actions
                self.actions_distribution = tf.squeeze(
                    tf.nn.softmax(self.output))
                # Loss with negative log probability
                self.neg_log_prob = softmax_cross_entropy_with_logits_v2(
                    logits=self.output, labels=self.actions_distribution)
                self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)

            self.vars_to_train = [policy_out.W1, policy_out.b1,
                                  policy_out.W2, policy_out.b2,
                                  ]
            self.optimizer = AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss, var_list=self.vars_to_train)
            self.merged = tf.compat.v1.summary.merge_all()


class ProgressiveAgent:
    def __init__(self, env_name, source_dict):
        self.env_name = source_dict["out"]
        self.source_dict = source_dict
        self.env = gym.make(self.env_name)
        self.convergence_treshold = ENV_TO_REWARD_THRESHOLD[env_name]
        self.original_action_size = ENV_TO_ACTION_SIZE[env_name]
        self.original_state_size = ENV_TO_STATE_SIZE[env_name]
        
    def _pad_state(self, state):
         return np.reshape(np.pad(state, (0, STATE_SIZE - self.original_state_size)),
                           [1, STATE_SIZE])

    def run(self, discount_factor, learning_rate_value,learning_rate,
        num_hidden_layers, num_neurons_value,input_1_parms,input_2_parms,out_parms):
        
        reset_default_graph()

        mountain_car = self.env_name == OpenGymEnvs.MOUNTAIN_CAR.value

        if mountain_car:
            state_space_samples = np.array(
                [self._pad_state(self.env.observation_space.sample())
                 for x in range(10000)]).reshape(10000, STATE_SIZE)
            scaler = StandardScaler()
            scaler.fit(state_space_samples)
        
        # Initialize the policy network

        self.policy_source1 = PolicyNetwork(
            input_1_parms["learning_rate"], input_1_parms["num_neurons_policy"],
            mountain_car=mountain_car,
            progressive=self.source_dict["input_1"])
        self.policy_source2 = PolicyNetwork(
            input_2_parms["learning_rate"], input_2_parms["num_neurons_policy"],
            mountain_car=mountain_car,
            progressive=self.source_dict["input_2"])

        self.policy_out = PolicyNetwork(
            learning_rate, out_parms["num_neurons_policy"],
            mountain_car=mountain_car,
            progressive=self.source_dict["out"])

        self.progressive_policy = ProgressivePolicyNetwork(learning_rate,
                                                           self.policy_source1,
                                                           self.policy_source2,
                                                           self.policy_out,
                                                           mountain_car=mountain_car)

        self.value_function = ValueNetwork(learning_rate_value, num_hidden_layers, num_neurons_value)

        saver_source1 = Saver(var_list=self.policy_source1.var_to_save_progressive)
        saver_source2 = Saver(var_list= self.policy_source2.var_to_save_progressive)

        with Session() as sess:
            sess.run(global_variables_initializer())

            saver_source1.restore(sess, f"/tmp/{self.source_dict['input_1']}-for-transfer-model.ckpt")
            saver_source2.restore(sess, f"/tmp/{self.source_dict['input_2']}-for-transfer-model.ckpt")

            # initiate log files
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = (f'../logs/gradient_tape/{self.env_name}/' +
                             current_time + '/train')
            train_summary_writer = tf.compat.v1.summary.FileWriter(train_log_dir, sess.graph)

            all_avg = []
            solved = False
            episode_rewards = np.zeros(MAX_EPISODES)
            average_rewards = -np.inf

            for episode in range(MAX_EPISODES):
                current_state = self.env.reset()
                
                current_state = self._pad_state(current_state)

                if mountain_car:
                       current_state = scale_state(scaler, current_state)
                       
                       
                I = 1
                while True:
                    actions_distribution = sess.run(self.progressive_policy.actions_distribution,
                                                    {self.policy_source1.state: current_state,
                                                     self.policy_source2.state: current_state,
                                                     self.policy_out.state: current_state})
                    if not mountain_car:
                        actions_distribution = actions_distribution[:self.original_action_size]
                        actions_distribution = actions_distribution / actions_distribution.sum()
                        action = np.random.choice(
                            np.arange(self.original_action_size), p=actions_distribution)
                    else:
                        action = [actions_distribution[0]]
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = self._pad_state(next_state)
 
                    if mountain_car:
                        next_state = scale_state(scaler, next_state)
                        
                        
                    episode_rewards[episode] += reward

                    current_state_prediction = sess.run([self.value_function.final_output],
                                                        {self.value_function.state: current_state,
                                                         self.value_function.total_discounted_return: reward})
                    next_state_prediction = sess.run([self.value_function.final_output],
                                                     {self.value_function.state: next_state,
                                                      self.value_function.total_discounted_return: reward})

                    reward_estemation = reward + \
                                        (1 - int(done)) * discount_factor * \
                                        next_state_prediction[0][0][0]
                    delta = reward_estemation - \
                            current_state_prediction[0][0][0]
                    delta *= I

                    sess.run([self.value_function.optimizer], {self.value_function.state: current_state,
                                                               self.value_function.total_discounted_return: reward_estemation,
                                                               self.value_function.delta: 1})

                    action_one_hot = np.zeros(ACTION_SIZE)
                    if mountain_car:
                        action_one_hot[0] = action[0]
                    else:
                        action_one_hot[action] = 1
                    feed_dict = {self.policy_source1.state: current_state,
                                 self.policy_source2.state: current_state,
                                 self.policy_out.state: current_state,
                                 self.progressive_policy.R_t: delta,
                                 self.progressive_policy.action: action_one_hot,
                                 self.progressive_policy.reward_per_episode: episode_rewards[episode],
                                 self.policy_source1.reward_per_episode: episode_rewards[episode],
                                 self.policy_source2.reward_per_episode: episode_rewards[episode],
                                 self.policy_out.reward_per_episode: episode_rewards[episode]}
                    _, summary = sess.run([self.progressive_policy.optimizer,
                                           self.progressive_policy.merged],
                                          feed_dict)

                    if done:
                        if episode > 98:
                            # Check if solved
                            average_rewards = np.mean(
                                episode_rewards[(episode - 99):episode + 1])
                        print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode,
                                                                                           episode_rewards[episode],
                                                                                           round(average_rewards, 2)))
                        all_avg.append(round(average_rewards, 2))
                        if average_rewards >= self.convergence_treshold:
                            print(' Solved at episode: ' + str(episode))
                            solved = True
                        break

                    current_state = next_state
                    I *= discount_factor

                if solved:
                    break

                train_summary_writer.add_summary(summary, episode)

        return max(all_avg)


def gridSearch(parmas, env_name,transfer_dict, nSearch=100, maxN=False):
    paramsList = list(ParameterGrid(parmas))
    shuffle(paramsList)

    #     if nSearch > len(paramsList) or maxN:
    #         nSearch = len(paramsList)

    gridSearchResults = []
    agent = ProgressiveAgent(env_name,transfer_dict)
    for paramsDict in tqdm(paramsList[:nSearch]):
        try:
            print(paramsDict)
            max_reward_avg = agent.run(**paramsDict)
            paramsDict['max_average_reward_100_episodes'] = max_reward_avg
            print(paramsDict)
            gridSearchResults.append(paramsDict)
        except Exception as e:
            print(e)
            print(paramsDict)
            continue

    hyperparameterTable = pd.DataFrame(gridSearchResults)
    hyperparameterTable.sort_values(
        "max_average_reward_100_episodes", inplace=True)
    hyperparameterTable.to_csv(f"HP-{env_name.value}.csv")
    print(hyperparameterTable)


def run_grid_search_transfer(params,env,transfer_dict):
    params = {"discount_factor": [0.99, 0.95],
              "learning_rate": [0.01, 0.001, 0.0001, 0.00001],
              "learning_rate_value": [0.01, 0.001, 0.0001, 0.00001],
              **params}
    gridSearch(params, env,transfer_dict)


    
def progressiveCART_ACRO__MOUBTAIN(grid_search = False):
    transfer_dict = {'input_1': OpenGymEnvs.CARTPOLE.value,
                  'input_2': OpenGymEnvs.ACROBOT.value,
                  'out': OpenGymEnvs.MOUNTAIN_CAR.value}
    agent = ProgressiveAgent(OpenGymEnvs.MOUNTAIN_CAR, transfer_dict)
    
    CARTPOLE_parms = getBestParamsCart()
    ACROBOT_parms = getBestParamsAcro()
    MOUNTAIN_CAR_parms = getBestParamsMountain()
    
    learning_rate = MOUNTAIN_CAR_parms["learning_rate"]
    
    if grid_search : 
        grid_params = {'num_hidden_layers': 2,
                  'num_neurons_value': 12, "input_1_parms" : CARTPOLE_parms,"input_2_parms" :ACROBOT_parms,
                  "out_parms" : MOUNTAIN_CAR_parms
                  }
    
        run_grid_search_transfer()
    else:
        parameters = { 'discount_factor': 0.99, 'learning_rate_value': 0.0004,'num_hidden_layers': 2,
                  'num_neurons_value': 12, "input_1_parms" : CARTPOLE_parms,"input_2_parms" :ACROBOT_parms,
                  "out_parms" : MOUNTAIN_CAR_parms,"learning_rate" : learning_rate
                  }
        agent.run(**parameters)
    
def progressiveACRO_MOUNTAIN__CART(grid_search = False):
    
    env_name = OpenGymEnvs.CARTPOLE
    transfer_dict = {'input_1': OpenGymEnvs.MOUNTAIN_CAR.value,
                  'input_2': OpenGymEnvs.ACROBOT.value,
                  'out': OpenGymEnvs.CARTPOLE.value}
    agent = ProgressiveAgent(env_name, transfer_dict)
    
    CARTPOLE_parms = getBestParamsCart()
    ACROBOT_parms = getBestParamsAcro()
    MOUNTAIN_CAR_parms = getBestParamsMountain()
    
    learning_rate = CARTPOLE_parms["learning_rate"]
    
    
    if grid_search : 
        grid_params = {'num_hidden_layers': [2,4,6,8,12],
                  'num_neurons_value': [100,10,50], "input_1_parms" : [MOUNTAIN_CAR_parms] ,"input_2_parms" :[ACROBOT_parms],
                  "out_parms" :[CARTPOLE_parms]
                  }
    
        run_grid_search_transfer(grid_params, env_name,transfer_dict)
    else:
        parameters = { 'discount_factor': 0.99, 'learning_rate_value': 0.0004,'num_hidden_layers': 2,
                  'num_neurons_value': 12, "input_1_parms" : CARTPOLE_parms,"input_2_parms" :ACROBOT_parms,
                  "out_parms" : MOUNTAIN_CAR_parms,"learning_rate" : learning_rate
                  }
        agent.run(**parameters)
        
        
if __name__ == '__main__':
   progressiveACRO_MOUNTAIN__CART(grid_search = True)  ## need to replace by section