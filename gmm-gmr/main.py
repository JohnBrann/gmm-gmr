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
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mixtures import GMM_GMR

def save_skill_to_h5(times, trajectory, folder_path="skills"):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Determine the next available file number in the folder
    existing_files = os.listdir(folder_path)
    pattern = r"skill_(\d+)\.h5"
    max_num = 0
    for filename in existing_files:
        m = re.match(pattern, filename)
        if m:
            num = int(m.group(1))
            if num > max_num:
                max_num = num
    new_file_num = max_num + 1
    new_file_name = f"skill_{new_file_num}.h5"
    full_path = os.path.join(folder_path, new_file_name)
    
    # Save the data to the HDF5 file
    with h5py.File(full_path, "w") as f:
        f.create_dataset("times", data=times)
        f.create_dataset("trajectory", data=trajectory)
    
    print(f"Skill saved to {full_path}")

# Loads demonstrations in h5 format.
# If the dataset_key is 'states', we assume the first column is time and skip it.
# For end-effector demonstrations (dataset_key 'eef_positions'), we assume data is (n_timesteps, 3).
def load_demonstrations(folder_path, dataset_key='eef_positions'):
    demonstrations = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.h5'):
            filepath = os.path.join(folder_path, filename)
            with h5py.File(filepath, 'r') as f:
                data = np.array(f[dataset_key])
            if dataset_key == 'states':
                data = data[:, 1:4]
            demonstrations.append(data)
            mid_idx = len(data) // 2
            print(f"Loaded {filename}: Shape {data.shape}")
            print(f"  Start: {data[:2]}")  # First 2 rows
            print(f"  Middle: {data[mid_idx-1:mid_idx+1]}")  # 2 rows from the middle
            print(f"  End: {data[-2:]}")  # Last 2 rows
    return demonstrations

# -------------------------------
# Main script
# -------------------------------

# Specify demonstration duration in seconds (e.g., 5 seconds)
demo_duration = 5.0

# Load the h5 demonstration files from the specified folder using end-effector positions.
# Adjust the folder path as needed.
demonstrations = load_demonstrations('../demonstration_collection/smoothed_demonstrations', dataset_key='eef_positions')

# Optionally, print the shape of each loaded demonstration for debugging
for i, demo in enumerate(demonstrations):
    print("Demonstration {} shape: {}".format(i, demo.shape))

# Instantiate and fit the GMM_GMR model using the processed demonstrations.
# The second argument (3) represents the 3 dimensions: [x, y, z].
gmm_gmr = GMM_GMR(demonstrations, 3, demo_duration=demo_duration)
gmm_gmr.fit()

# Set up subplots for a 3-dimensional demonstration (x, y, z)
fig, axarr = plt.subplots(3, 1, figsize=(8, 12))

# For each demonstration, create a time vector that spans from 0 to demo_duration seconds.
for i in range(len(gmm_gmr.trajectories)):
    T = gmm_gmr.trajectories[i].shape[0]
    time_vec = np.linspace(0, demo_duration, T)
    for j in range(3):
        axarr[j].plot(time_vec,
                      gmm_gmr.trajectories[i, :, j],
                      linestyle=':',
                      label='Demo {}'.format(i) if i == 0 else "")

# Overlay the centers from the GMM (which are now scaled in seconds)
for j in range(3):
    axarr[j].scatter(gmm_gmr.centers_temporal,
                     gmm_gmr.centers_spatial[:, j],
                     label='centers')

# Generate an estimated trajectory using GMR and plot it.
# The generated times are now in seconds.
times, trj = gmm_gmr.generate_trajectory(0.1)
save_skill_to_h5(times, trj)

for j in range(3):
    axarr[j].plot(times, trj[:, j], label='estimated')

# Set labels and title for the plots
axarr[0].set_ylabel('X')
axarr[1].set_ylabel('Y')
axarr[2].set_ylabel('Z')
axarr[2].set_xlabel('Time (s)')
axarr[0].set_title('GMM-GMR: Demonstrations & Estimated Trajectory (EEF)')

# Add legends and show the plot
for ax in axarr:
    ax.legend()

# Save the figure as an image file
plt.savefig('gmm_gmr_result.png')
plt.show()


# # import os
# # import re
# # import h5py
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from mixtures import GMM_GMR


# # def save_skill_to_h5(times, trajectory, folder_path="skills"):
# #     # Create the folder if it doesn't exist
# #     if not os.path.exists(folder_path):
# #         os.makedirs(folder_path)
    
# #     # Determine the next available file number in the folder
# #     existing_files = os.listdir(folder_path)
# #     pattern = r"skill_(\d+)\.h5"
# #     max_num = 0
# #     for filename in existing_files:
# #         m = re.match(pattern, filename)
# #         if m:
# #             num = int(m.group(1))
# #             if num > max_num:
# #                 max_num = num
# #     new_file_num = max_num + 1
# #     new_file_name = f"skill_{new_file_num}.h5"
# #     full_path = os.path.join(folder_path, new_file_name)
    
# #     # Save the data to the HDF5 file
# #     with h5py.File(full_path, "w") as f:
# #         f.create_dataset("times", data=times)
# #         f.create_dataset("trajectory", data=trajectory)
    
# #     print(f"Skill saved to {full_path}")

# # # Loads demonstrations in h5 format
# # def load_demonstrations(folder_path, dataset_key='states'):
# #     demonstrations = []
# #     for filename in sorted(os.listdir(folder_path)):
# #         if filename.endswith('.h5'):
# #             filepath = os.path.join(folder_path, filename)
# #             with h5py.File(filepath, 'r') as f:
# #                 data = np.array(f[dataset_key])
# #             # Skip the first column (time) and keep columns 1-3 (x, y, z)
# #             data = data[:, 1:4]
# #             demonstrations.append(data)
# #     return demonstrations

# # # Load the h5 demonstration files from the specified folder
# # demonstrations = load_demonstrations('../demonstration_collection/smoothed_demonstrations', dataset_key='states')

# # # Optionally, print the shape of each loaded demonstration for debugging
# # for i, demo in enumerate(demonstrations):
# #     print("Demonstration {} shape: {}".format(i, demo.shape))

# # # Instantiate and fit the GMM_GMR model using the processed demonstrations
# # gmm_gmr = GMM_GMR(demonstrations, 3)
# # gmm_gmr.fit()

# # # Set up subplots for a 3-dimensional demonstration (x, y, z)
# # fig, axarr = plt.subplots(3, 1, figsize=(8, 12))

# # # Plot each original demonstration (using a dotted line)
# # for i in range(len(gmm_gmr.trajectories)):
# #     for j in range(3):
# #         axarr[j].plot(gmm_gmr.trajectories[i, :, j],
# #                       linestyle=':',
# #                       label='Demo {}'.format(i) if i == 0 else "")

# # # Overlay the centers from the GMM (plotted against the generated time steps)
# # for j in range(3):
# #     axarr[j].scatter(gmm_gmr.centers_temporal,
# #                      gmm_gmr.centers_spatial[:, j],
# #                      label='centers')

# # # Generate an estimated trajectory using GMR and plot it
# # times, trj = gmm_gmr.generate_trajectory(0.1)

# # # save the generated trajectory to an h5 file
# # save_skill_to_h5(times, trj)

# # for j in range(3):
# #     axarr[j].plot(times, trj[:, j], label='estimated')

# # # Add legends and show the plot
# # for ax in axarr:
# #     ax.legend()

# # # Save the figure as an image file
# # plt.savefig('gmm_gmr_result.png')

# # plt.show()

# import os
# import re
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# from mixtures import GMM_GMR

# def save_skill_to_h5(times, trajectory, folder_path="skills"):
#     # Create the folder if it doesn't exist
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
    
#     # Determine the next available file number in the folder
#     existing_files = os.listdir(folder_path)
#     pattern = r"skill_(\d+)\.h5"
#     max_num = 0
#     for filename in existing_files:
#         m = re.match(pattern, filename)
#         if m:
#             num = int(m.group(1))
#             if num > max_num:
#                 max_num = num
#     new_file_num = max_num + 1
#     new_file_name = f"skill_{new_file_num}.h5"
#     full_path = os.path.join(folder_path, new_file_name)
    
#     # Save the data to the HDF5 file
#     with h5py.File(full_path, "w") as f:
#         f.create_dataset("times", data=times)
#         f.create_dataset("trajectory", data=trajectory)
    
#     print(f"Skill saved to {full_path}")

# # Loads demonstrations in h5 format
# def load_demonstrations(folder_path, dataset_key='states'):
#     demonstrations = []
#     for filename in sorted(os.listdir(folder_path)):
#         if filename.endswith('.h5'):
#             filepath = os.path.join(folder_path, filename)
#             with h5py.File(filepath, 'r') as f:
#                 data = np.array(f[dataset_key])
#             # Skip the first column (time) and keep columns 1-3 (x, y, z)
#             data = data[:, 1:4]
#             demonstrations.append(data)
#             mid_idx = len(data) // 2
#             print(f"Loaded {filename}: Shape {data.shape}")
#             print(f"  Start: {data[:2]}")  # First 2 rows
#             print(f"  Middle: {data[mid_idx-1:mid_idx+1]}")  # 2 rows from the middle
#             print(f"  End: {data[-2:]}")  # Last 2 rows
#     return demonstrations

# # -------------------------------
# # Main script
# # -------------------------------

# # Specify demonstration duration in seconds (e.g., 5 seconds)
# demo_duration = 5.0

# # Load the h5 demonstration files from the specified folder
# demonstrations = load_demonstrations('../demonstration_collection/smoothed_demonstrations', dataset_key='states')

# # Optionally, print the shape of each loaded demonstration for debugging
# for i, demo in enumerate(demonstrations):
#     print("Demonstration {} shape: {}".format(i, demo.shape))

# # Instantiate and fit the GMM_GMR model using the processed demonstrations.
# # Pass the demo_duration to scale the temporal axis.
# gmm_gmr = GMM_GMR(demonstrations, 3, demo_duration=demo_duration)
# gmm_gmr.fit()

# # Set up subplots for a 3-dimensional demonstration (x, y, z)
# fig, axarr = plt.subplots(3, 1, figsize=(8, 12))

# # For each demonstration, create a time vector that spans from 0 to demo_duration seconds.
# for i in range(len(gmm_gmr.trajectories)):
#     T = gmm_gmr.trajectories[i].shape[0]
#     time_vec = np.linspace(0, demo_duration, T)
#     for j in range(3):
#         axarr[j].plot(time_vec,
#                       gmm_gmr.trajectories[i, :, j],
#                       linestyle=':',
#                       label='Demo {}'.format(i) if i == 0 else "")

# # Overlay the centers from the GMM (which are now scaled in seconds)
# for j in range(3):
#     axarr[j].scatter(gmm_gmr.centers_temporal,
#                      gmm_gmr.centers_spatial[:, j],
#                      label='centers')

# # Generate an estimated trajectory using GMR and plot it.
# # The generated times are now in seconds.
# times, trj = gmm_gmr.generate_trajectory(0.1)
# save_skill_to_h5(times, trj)

# for j in range(3):
#     axarr[j].plot(times, trj[:, j], label='estimated')

# # Set labels and title
# axarr[0].set_ylabel('X')
# axarr[1].set_ylabel('Y')
# axarr[2].set_ylabel('Z')
# axarr[2].set_xlabel('Time (s)')
# axarr[0].set_title('GMM-GMR: Demonstrations & Estimated Trajectory')

# # Add legends and show the plot
# for ax in axarr:
#     ax.legend()

# # Save the figure as an image file
# plt.savefig('gmm_gmr_result.png')
# plt.show()

