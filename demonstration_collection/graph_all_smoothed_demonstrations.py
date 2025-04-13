import os
import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  


folder_path = "smoothed_demonstrations"
plots_dir = "plots"

if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Get all files in smoothed_demonstration folder
files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]

# 3D demonstration plot
fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(111, projection='3d')

colors = plt.cm.hsv(np.linspace(0, 1, len(files) + math.ceil(0.8 * len(files))))

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
ax_3d.set_title('3D End-Effector Trajectories from All Smoothed Demonstrations')
ax_3d.legend()

fig_3d.savefig(os.path.join(plots_dir, "all_smoothed_demos_3d.png"))

# 2D plots for x, y and z end effector positions, as well as grip strength
fig_2d, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 9), sharex=True)

axs[0].set_ylabel('X')
axs[1].set_ylabel('Y')
axs[2].set_ylabel('Z')
axs[3].set_ylabel('Grip Strength')
axs[3].set_xlabel('Time (s)')

for idx, file in enumerate(files):
    file_path = os.path.join(folder_path, file)
    with h5py.File(file_path, "r") as f:
        timestamps = np.array(f["timestamps"])
        positions = np.array(f["eef_positions"])
        grip_strength = np.array(f["grip_strength"])

    # Plot X, Y, Z, and grip strength vs. time in corresponding subplots
    axs[0].plot(timestamps, positions[:, 0], color=colors[idx], label=file)
    axs[1].plot(timestamps, positions[:, 1], color=colors[idx], label=file)
    axs[2].plot(timestamps, positions[:, 2], color=colors[idx], label=file)
    axs[3].plot(timestamps, grip_strength, color=colors[idx], label=file)

axs[0].set_title('Smoothed End-Effector Demonstrations Over Time (X, Y, Z)')
axs[0].legend()

fig_2d.savefig(os.path.join(plots_dir, "all_smoothed_demos_2d.png"))
plt.show()
