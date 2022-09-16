import CA

def right_control(observation):
    return 1


def random_control(observation):
    import random

    action = random.randint(0, 1)
    return action


def naive_control(observation):
    if observation[2] >= 0:
        return 1
    else:
        return 0


def simple_ca(observation, rule):
    state = CA.observables_to_binary(observation)[1:]
    action = CA.increment_state(state, rule)

    return action


def voter_control(observation):
    state = CA.observables_to_binary(observation)
    sum = 0
    for i in state:
        sum += int(i)
    print(sum)
    if sum/4 < 1/2:
        return 1
    else:
        return 0
