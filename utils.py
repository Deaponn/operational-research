import numpy as np

def normalize(transmitters, radius):
    biggest = np.max(transmitters)
    return transmitters / biggest, radius / biggest
