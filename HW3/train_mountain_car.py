from HW3.actor_critic import Agent
from HW3.actor_critic import OpenGymEnvs


def transfer_from_cartpole():
    agent = Agent(OpenGymEnvs.MOUNTAIN_CAR)
    best_parameters = {'discount_factor': 0.99, 'learning_rate': 0.0001, 'learning_rate_value': 0.001,
                       'num_hidden_layers': 3, 'num_neurons_value': 64, 'num_neurons_policy': 12}
    ret = agent.run(**best_parameters, restore_sess=OpenGymEnvs.CARTPOLE.value)


def train_mountain_car():
    agent = Agent(OpenGymEnvs.MOUNTAIN_CAR)
    best_parameters = {'discount_factor': 0.99, 'learning_rate': 0.00001, 'learning_rate_value': 0.0004,
                       'num_hidden_layers': 2, 'num_neurons_value': 12, 'num_neurons_policy': 12}
    ret = agent.run(**best_parameters)


if __name__ == '__main__':
    train_mountain_car()
    # transfer_from_acrobot()
    # run_grid_search_transfer(OpenGymEnvs.CARTPOLE, OpenGymEnvs.ACROBOT.value,
    #                          {'num_hidden_layers': [3], 'num_neurons_value': [64], 'num_neurons_policy': [12]})
