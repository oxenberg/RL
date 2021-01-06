from HW3.actor_critic import Agent
from HW3.actor_critic import OpenGymEnvs

if __name__ == '__main__':
    agent = Agent(OpenGymEnvs.ACROBOT)
    # 0.95, 0.001, 0.0001, 3, 64
    best_parameters = {'discount_factor': 0.95, 'learning_rate': 0.001, 'learning_rate_value': 0.0001,
                       'num_hidden_layers': 3, 'num_neurons_value': 64, 'num_neurons_policy': 12}
    ret = agent.run(**best_parameters)
