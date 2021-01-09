from actor_critic import Agent
from actor_critic import OpenGymEnvs


def transfer_from_acrobot():
    agent = Agent(OpenGymEnvs.CARTPOLE)
    best_parameters = {'discount_factor': 0.99, 'learning_rate': 0.0001, 'learning_rate_value': 0.001,
                       'num_hidden_layers': 3, 'num_neurons_value': 64, 'num_neurons_policy': 12}
    ret = agent.run(**best_parameters, restore_sess=OpenGymEnvs.ACROBOT.value)


def train_cartpole():
    agent = Agent(OpenGymEnvs.CARTPOLE)
    best_parameters = {'discount_factor': 0.99, 'learning_rate': 0.0001, 'learning_rate_value': 0.001,
                       'num_hidden_layers': 2, 'num_neurons_value': 64, 'num_neurons_policy': 12}
    ret = agent.run(**best_parameters)

def train_cartpole_progressive():
    agent = Agent(OpenGymEnvs.CARTPOLE)
    best_parameters = {'discount_factor': 0.99, 'learning_rate': 0.0001, 'learning_rate_value': 0.001,
                       'num_hidden_layers': 2, 'num_neurons_value': 64, 'num_neurons_policy': 12}
    ret = agent.run(**best_parameters,for_transfer=True)
if __name__ == '__main__':
    # train_cartpole()
    train_cartpole_progressive()
    # run_grid_search_transfer(OpenGymEnvs.CARTPOLE, OpenGymEnvs.ACROBOT.value,
    #                          {'num_hidden_layers': [3], 'num_neurons_value': [64], 'num_neurons_policy': [12]})
