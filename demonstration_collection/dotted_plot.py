import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Load raw data from the HDF5 file
with h5py.File("smoothed_demo_4.h5", "r") as f:
    timestamps = np.array(f["timestamps"])
    states = np.array(f["states"])  # Assume shape (n_timesteps, state_dim)

# Extract the first three elements assuming [x, y, z] positions
raw_trajectory = states[:, :3]

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot each point in 3D space
ax.scatter(
    raw_trajectory[:, 0],
    raw_trajectory[:, 1],
    raw_trajectory[:, 2],
    marker='o',
    s=20,        # Marker size (adjust as needed)
    alpha=0.8,   # Marker transparency
    c='blue'     # Color (or you can map this to time)
)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Demonstration Trajectory (Scatter)')

plt.show()
