from actor_critic import Agent
from actor_critic import OpenGymEnvs
import time


def getBestParamsCart():
    return {"learning_rate" : 0.0001,
                     "num_neurons_policy" : 12}

def createParams():
    genral_policy_params = getBestParamsCart()
    spefice_params = {'discount_factor': 0.99,'learning_rate_value': 0.001,'num_neurons_value': 64,
                      'num_hidden_layers': 2}
    
    all_params = {}
    #concatinate params
    for d in (genral_policy_params, spefice_params): 
        all_params.update(d)
        
    return all_params


def transfer_from_acrobot():
    agent = Agent(OpenGymEnvs.CARTPOLE)
    best_parameters = {'discount_factor': 0.99, 'learning_rate': 0.0001, 'learning_rate_value': 0.001,
                       'num_hidden_layers': 3, 'num_neurons_value': 64, 'num_neurons_policy': 12}
    _ = agent.run(**best_parameters, restore_sess=OpenGymEnvs.ACROBOT.value)


def train_cartpole():
    agent = Agent(OpenGymEnvs.CARTPOLE)
    best_parameters = {'discount_factor': 0.99, 'learning_rate': 0.0001, 'learning_rate_value': 0.001,
                       'num_hidden_layers': 2, 'num_neurons_value': 64, 'num_neurons_policy': 12}
    _ = agent.run(**best_parameters)

def train_cartpole_progressive():
    agent = Agent(OpenGymEnvs.CARTPOLE)
    best_parameters = createParams()
    _ = agent.run(**best_parameters,for_transfer=True)
    
    
if __name__ == '__main__':
    train_cartpole_progressive() ## need to replace by section
