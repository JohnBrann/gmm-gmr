import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  


folder_path = "smoothed_demonstrations"

# Get all files in smoothed_demonstration folder
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
ax_3d.set_title('3D End-Effector Trajectories from All Smoothed Demonstrations')
ax_3d.legend()

fig_3d.savefig("all_smoothed_demos_3d.png")

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

axs[0].set_title('Smoothed End-Effector Demonstrations Over Time (X, Y, Z)')
axs[0].legend()

fig_2d.savefig("all_smoothed_demos_2d.png")
plt.show()


# import os
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # Registers the 3D projection

# # Folder containing smoothed demonstration HDF5 files
# folder_path = "smoothed_demonstrations"

# # Get a list of all .h5 files in the folder
# files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]

# # -------------------------------------------------
# # 1) Create a 3D plot for all demonstrations
# # -------------------------------------------------
# fig_3d = plt.figure()
# ax_3d = fig_3d.add_subplot(111, projection='3d')

# # Generate a unique color for each demonstration (up to 10 distinct colors here)
# colors = plt.cm.tab10(np.linspace(0, 1, len(files)))

# for idx, file in enumerate(files):
#     file_path = os.path.join(folder_path, file)
#     with h5py.File(file_path, "r") as f:
#         # Load timestamps and states
#         timestamps = np.array(f["timestamps"])
#         states = np.array(f["states"])  # shape: (n_timesteps, state_dim)

#     # Skip the first column (time) and take the next three elements as [x, y, z]
#     raw_trajectory = states[:, 1:4]

#     # Plot the 3D trajectory
#     ax_3d.plot(
#         raw_trajectory[:, 0],
#         raw_trajectory[:, 1],
#         raw_trajectory[:, 2],
#         color=colors[idx],
#         label=file
#     )

# ax_3d.set_xlabel('X')
# ax_3d.set_ylabel('Y')
# ax_3d.set_zlabel('Z')
# ax_3d.set_title('3D Trajectories from All Demonstrations')
# ax_3d.legend()

# # Save the 3D figure
# fig_3d.savefig("all_smoothed_demos_3d.png")

# # -------------------------------------------------
# # 2) Create 2D subplots for X, Y, and Z vs. Time
# # -------------------------------------------------
# fig_2d, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 9), sharex=True)

# # Set axis labels for each subplot
# axs[0].set_ylabel('X')
# axs[1].set_ylabel('Y')
# axs[2].set_ylabel('Z')
# axs[2].set_xlabel('Time (s)')

# for idx, file in enumerate(files):
#     file_path = os.path.join(folder_path, file)
#     with h5py.File(file_path, "r") as f:
#         timestamps = np.array(f["timestamps"])
#         states = np.array(f["states"])

#     # Skip the first column (time) and take the next three as [x, y, z]
#     raw_trajectory = states[:, 1:4]

#     # Plot X, Y, and Z vs. time in corresponding subplots
#     axs[0].plot(timestamps, raw_trajectory[:, 0], color=colors[idx], label=file)
#     axs[1].plot(timestamps, raw_trajectory[:, 1], color=colors[idx], label=file)
#     axs[2].plot(timestamps, raw_trajectory[:, 2], color=colors[idx], label=file)

# axs[0].set_title('Smoothed Demonstrations Over Time (X, Y, Z)')

# # Optionally, show a legend in the top subplot
# axs[0].legend()

# # Save the 2D figure
# fig_2d.savefig("all_smoothed_demos_2d.png")

# plt.show()
