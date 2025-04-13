"""
This module uses code adapted from the GMM-GMR implementation available at:
    https://github.com/ceteke/GMM-GMR
which is based on the following paper:
    Calinon, S., Guenter, F., & Billard, A. (2007). 
    "On Learning, Representing, and Generalizing a Task in a Humanoid Robot".
    IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics).
    Available at: https://ieeexplore.ieee.org/document/4126276/

This adaptation is provided under the same licensing terms as the original repository.
"""

from dtw import dtw
import numpy as np
import math

def align_trajectories(data):
    ls = np.argmax([d.shape[0] for d in data])

    data_warp = []

    for d in data:
        dist, cost, acc, path = dtw(data[ls], d,
                                    dist=lambda x, y: np.linalg.norm(x - y, ord=1))

        data_warp += [d[path[1]][:data[ls].shape[0]]]

    return data_warp

def gaussian(x, mu, var):
    exponent = -((x - mu) ** 2) / (2 * var)
    return (1 / math.sqrt((2 * math.pi * var))) * math.exp(exponent)
