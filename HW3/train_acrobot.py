from actor_critic import Agent
from actor_critic import OpenGymEnvs


def getBestParamsAcro():
    return {"learning_rate": 0.001,
            "num_neurons_policy": 12}


def createParams():
    genral_policy_params = getBestParamsAcro()
    spefice_params = {'discount_factor': 0.95, 'learning_rate_value': 0.0001, 'num_neurons_value': 64,
                      'num_hidden_layers': 3}

    all_params = {}
    # concatinate params
    for d in (genral_policy_params, spefice_params):
        all_params.update(d)

    return all_params


def trainAcrobot():
    agent = Agent(OpenGymEnvs.ACROBOT)
    best_parameters = createParams()
    _ = agent.run(**best_parameters)


def trainAcrobotForProgressive():
    agent = Agent(OpenGymEnvs.ACROBOT)
    best_parameters = createParams()
    _ = agent.run(**best_parameters, for_transfer=True)


if __name__ == '__main__':
    trainAcrobotForProgressive()  ## need to replace by section
