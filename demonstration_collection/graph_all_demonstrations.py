import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

folder_path = "demonstrations"

# Get all files in demonstration folder
files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]

# 3d demonstration plot
fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(111, projection='3d')


colors = plt.cm.tab10(np.linspace(0, 1, len(files)))

for idx, file in enumerate(files):
    file_path = os.path.join(folder_path, file)
    with h5py.File(file_path, "r") as f:
        timestamps = np.array(f["timestamps"])
        positions = np.array(f["eef_positions"]) 
    # Plot the 3D trajectory
    ax_3d.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color=colors[idx],
        label=file
    )

ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')
ax_3d.set_title('3D End-Effector Trajectories from All Demonstrations')
ax_3d.legend()

fig_3d.savefig("all_demos_3d.png")

# 2D plots for x, y and z end effector positions
fig_2d, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 9), sharex=True)

axs[0].set_ylabel('X')
axs[1].set_ylabel('Y')
axs[2].set_ylabel('Z')
axs[2].set_xlabel('Time (s)')

for idx, file in enumerate(files):
    file_path = os.path.join(folder_path, file)
    with h5py.File(file_path, "r") as f:
        timestamps = np.array(f["timestamps"])
        positions = np.array(f["eef_positions"])

    # plot X, Y, and Z vs. time in corresponding subplots
    axs[0].plot(timestamps, positions[:, 0], color=colors[idx], label=file)
    axs[1].plot(timestamps, positions[:, 1], color=colors[idx], label=file)
    axs[2].plot(timestamps, positions[:, 2], color=colors[idx], label=file)

axs[0].set_title('End-Effector Demonstrations Over Time (X, Y, and Z)')
axs[0].legend()

fig_2d.savefig("all_demos_2d.png")
plt.show()

