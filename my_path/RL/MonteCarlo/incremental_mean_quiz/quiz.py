import numpy as np

def running_mean(x):
    mu = 0
    mean_values = []
    for k in np.arange(0, len(x)):
        # TODO: fill in the update step
        if k == 0:
            mu = x[k]
        else:
            mu = mu + (1/(k + 1)) * (x[k] - mu)

        mean_values.append(mu)
    return mean_values
