from actor_critic import Agent
from actor_critic import OpenGymEnvs


def transfer_from_acrobot():
    agent = Agent(OpenGymEnvs.CARTPOLE)
    best_parameters = {'discount_factor': 0.95, 'learning_rate': 0.001, 'learning_rate_value': 0.0001,
                       'num_hidden_layers': 3, 'num_neurons': 64}
    ret = agent.run(**best_parameters, restore_sess=OpenGymEnvs.ACROBOT.value)


def train_cartpole():
    agent = Agent(OpenGymEnvs.CARTPOLE)
    best_parameters = {'discount_factor': 0.99, 'learning_rate': 0.0001, 'learning_rate_value': 0.001,
                       'num_hidden_layers': 2, 'num_neurons': 64}
    ret = agent.run(**best_parameters)


if __name__ == '__main__':
    train_cartpole()
    transfer_from_acrobot()
