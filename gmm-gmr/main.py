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

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mixtures import GMM_GMR

# Loads demonstrations in h5 format
def load_demonstrations(folder_path, dataset_key='states'):
    demonstrations = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.h5'):
            filepath = os.path.join(folder_path, filename)
            with h5py.File(filepath, 'r') as f:
                data = np.array(f[dataset_key])
            # Skip the first column (time) and keep columns 1-3 (x, y, z)
            data = data[:, 1:4]
            demonstrations.append(data)
    return demonstrations

# Load the h5 demonstration files from the specified folder
demonstrations = load_demonstrations('../demonstration_collection/smoothed_demonstrations', dataset_key='states')

# Optionally, print the shape of each loaded demonstration for debugging
for i, demo in enumerate(demonstrations):
    print("Demonstration {} shape: {}".format(i, demo.shape))

# Instantiate and fit the GMM_GMR model using the processed demonstrations
gmm_gmr = GMM_GMR(demonstrations, 3)
gmm_gmr.fit()

# Set up subplots for a 3-dimensional demonstration (x, y, z)
fig, axarr = plt.subplots(3, 1, figsize=(8, 12))

# Plot each original demonstration (using a dotted line)
for i in range(len(gmm_gmr.trajectories)):
    for j in range(3):
        axarr[j].plot(gmm_gmr.trajectories[i, :, j],
                      linestyle=':',
                      label='Demo {}'.format(i) if i == 0 else "")

# Overlay the centers from the GMM (plotted against the generated time steps)
for j in range(3):
    axarr[j].scatter(gmm_gmr.centers_temporal,
                     gmm_gmr.centers_spatial[:, j],
                     label='centers')

# Generate an estimated trajectory using GMR and plot it
times, trj = gmm_gmr.generate_trajectory(0.1)
for j in range(3):
    axarr[j].plot(times, trj[:, j], label='estimated')

# Add legends and show the plot
for ax in axarr:
    ax.legend()

# Save the figure as an image file
plt.savefig('gmm_gmr_result.png')

plt.show()
