from HW3.actor_critic import Agent
from HW3.actor_critic import OpenGymEnvs
from HW3.actor_critic import run_grid_search_transfer


def transfer_from_acrobot():
    ## compare to 221656
    agent = Agent(OpenGymEnvs.CARTPOLE)
    best_parameters = {'discount_factor': 0.99, 'learning_rate': 0.0001, 'learning_rate_value': 0.001,
                       'num_hidden_layers': 3, 'num_neurons_value': 64, 'num_neurons_policy': 12}
    ret = agent.run(**best_parameters, restore_sess=OpenGymEnvs.ACROBOT.value)


def train_cartpole():
    agent = Agent(OpenGymEnvs.CARTPOLE)
    best_parameters = {'discount_factor': 0.99, 'learning_rate': 0.0001, 'learning_rate_value': 0.001,
                       'num_hidden_layers': 2, 'num_neurons_value': 64, 'num_neurons_policy': 12}
    ret = agent.run(**best_parameters)


if __name__ == '__main__':
    # train_cartpole()
    # transfer_from_acrobot()
    run_grid_search_transfer(OpenGymEnvs.CARTPOLE, OpenGymEnvs.ACROBOT.value,
                             {'num_hidden_layers': [3], 'num_neurons_value': [64], 'num_neurons_policy': [12]})
