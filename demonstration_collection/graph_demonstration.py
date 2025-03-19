# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Load raw data from the HDF5 file
# with h5py.File("demo.h5", "r") as f:
#     timestamps = np.array(f["timestamps"])
#     states = np.array(f["states"])  # Assume shape (n_timesteps, state_dim)

# # Extract the first three elements assuming [x, y, z] positions
# raw_trajectory = states[:, :3]

# # Create a 3D plot with color coding by time
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Use scatter with a colormap to represent time
# sc = ax.scatter(raw_trajectory[:, 0], raw_trajectory[:, 1], raw_trajectory[:, 2],
#                 c=timestamps, cmap='viridis', marker='o')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Trajectory (Colored by Time)')
# fig.colorbar(sc, label='Time (s)')

# plt.show()
# # plt.savefig()

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load raw data from the HDF5 file
with h5py.File("demo_2.h5", "r") as f:
    timestamps = np.array(f["timestamps"])
    states = np.array(f["states"])  # Assume shape: (n_timesteps, state_dim)

# Extract the first three elements assuming [x, y, z] positions
raw_trajectory = states[:, :3]

# ---------------------------
# 1) 3D Scatter Plot
# ---------------------------
fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(111, projection='3d')

sc = ax_3d.scatter(
    raw_trajectory[:, 0],
    raw_trajectory[:, 1],
    raw_trajectory[:, 2],
    c=timestamps,
    cmap='viridis',
    marker='o'
)
ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')
ax_3d.set_title('3D Trajectory (Colored by Time)')
fig_3d.colorbar(sc, label='Time (s)')

# ---------------------------
# 2) Time-Series Subplots
# ---------------------------
fig_2d, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 8), sharex=True)

# X vs. Time
axs[0].plot(timestamps, raw_trajectory[:, 0], color='r')
axs[0].set_ylabel('X')

# Y vs. Time
axs[1].plot(timestamps, raw_trajectory[:, 1], color='g')
axs[1].set_ylabel('Y')

# Z vs. Time
axs[2].plot(timestamps, raw_trajectory[:, 2], color='b')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Z')

fig_2d.suptitle('Trajectory vs. Time')

plt.show()


