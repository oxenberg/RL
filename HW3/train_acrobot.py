from actor_critic import Agent
from actor_critic import OpenGymEnvs



def trainAcrobotForProgressive(agent,best_parameters):
    ret = agent.run(**best_parameters, for_transfer=True)

def trainAcrobot(agent,best_parameters):
    ret = agent.run(**best_parameters, for_transfer=False)


if __name__ == '__main__':
    agent = Agent(OpenGymEnvs.ACROBOT)
    best_parameters = {'discount_factor': 0.95, 'learning_rate': 0.001, 'learning_rate_value': 0.0001,
                       'num_hidden_layers': 3, 'num_neurons_value': 64, 'num_neurons_policy': 12}
    trainAcrobotForProgressive(agent,best_parameters)
