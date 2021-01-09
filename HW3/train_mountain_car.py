from actor_critic import Agent
from actor_critic import OpenGymEnvs



def getBestParamsMountain():
    return {"learning_rate" : 0.0002,
                     "num_neurons_policy" : 40}

def createParams():
    genral_policy_params = getBestParamsMountain()
    spefice_params = {'discount_factor': 0.99,'learning_rate_value': 0.001,'num_neurons_value': 400,
                      'num_hidden_layers': 2}
    
    all_params = {}
    #concatinate params
    for d in (genral_policy_params, spefice_params): 
        all_params.update(d)
        
    return all_params

def transfer_from_cartpole():
    agent = Agent(OpenGymEnvs.MOUNTAIN_CAR)
    best_parameters = {'discount_factor': 0.99, 'learning_rate': 0.0001, 'learning_rate_value': 0.001,
                       'num_hidden_layers': 3, 'num_neurons_value': 64, 'num_neurons_policy': 12}
    ret = agent.run(**best_parameters, restore_sess=OpenGymEnvs.CARTPOLE.value)


def train_mountain_car():
    agent = Agent(OpenGymEnvs.MOUNTAIN_CAR)
    best_parameters = {'discount_factor': 0.99, 'learning_rate': 0.00002, 'learning_rate_value': 0.001,
                       'num_hidden_layers': 2, 'num_neurons_value': 400, 'num_neurons_policy': 40}
    ret = agent.run(**best_parameters)

def train_mountain_car_progressive():
    agent = Agent(OpenGymEnvs.MOUNTAIN_CAR)
    best_parameters = createParams()
    ret = agent.run(**best_parameters,for_transfer=True)

if __name__ == '__main__':
    train_mountain_car_progressive() ## need to replace by section
