import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Registers the 3D projection

# Folder containing demonstration HDF5 files
folder_path = "smoothed_demonstrations"

# Get a list of all .h5 files in the folder
files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]

# -------------------------------------------------
# 1) Create a 3D plot for all demonstrations
# -------------------------------------------------
fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(111, projection='3d')

# Generate a unique color for each demonstration (up to 10 distinct colors here)
colors = plt.cm.tab10(np.linspace(0, 1, len(files)))

for idx, file in enumerate(files):
    file_path = os.path.join(folder_path, file)
    with h5py.File(file_path, "r") as f:
        # Load timestamps and states
        timestamps = np.array(f["timestamps"])
        states = np.array(f["states"])  # shape: (n_timesteps, state_dim)

    # Extract the first three elements assuming [x, y, z] positions
    raw_trajectory = states[:, :3]

    # Plot the 3D trajectory
    ax_3d.plot(
        raw_trajectory[:, 0],
        raw_trajectory[:, 1],
        raw_trajectory[:, 2],
        color=colors[idx],
        label=file
    )

ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')
ax_3d.set_title('3D Trajectories from All Demonstrations')
ax_3d.legend()

# Save the 3D figure
fig_3d.savefig("all_smoothed_demos_3d.png")

# -------------------------------------------------
# 2) Create 2D subplots for Y and Z vs. Time
# -------------------------------------------------
fig_2d, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)

# Set axis labels for each subplot
axs[0].set_ylabel('Y')
axs[1].set_ylabel('Z')
axs[1].set_xlabel('Time (s)')

for idx, file in enumerate(files):
    file_path = os.path.join(folder_path, file)
    with h5py.File(file_path, "r") as f:
        timestamps = np.array(f["timestamps"])
        states = np.array(f["states"])

    raw_trajectory = states[:, :3]

    # Plot Y vs. time
    axs[0].plot(timestamps, raw_trajectory[:, 1], color=colors[idx], label=file)
    # Plot Z vs. time
    axs[1].plot(timestamps, raw_trajectory[:, 2], color=colors[idx], label=file)

axs[0].set_title('Smoothed Demonstrations Over Time (Y and Z Only)')

# Show a legend in the top subplot (optional)
axs[0].legend()

# Save the 2D figure
fig_2d.savefig("all_smoothed_demos_2d.png")

plt.show()
