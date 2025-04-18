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

plots_dir = "plots"

if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

def save_skill_to_h5(times, trajectory, gripper_strength, folder_path="skills"):
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
    
    # Save data
    with h5py.File(full_path, "w") as f:
        f.create_dataset("times", data=times)
        f.create_dataset("trajectory", data=trajectory)
        f.create_dataset("grip_strength", data=gripper_strength)
    
    print(f"Skill saved to {full_path}")

# Loads demonstrations in h5 format
def load_demonstrations(folder_path, dataset_key='eef_positions'):
    demonstrations = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.h5'):
            filepath = os.path.join(folder_path, filename)
            with h5py.File(filepath, 'r') as f:
                data = np.array(f[dataset_key])
                if data.ndim < 2:
                    data = data.reshape(-1, 1)
                 # For gripper_strength data, add a dummy extra dimension if needed.
                if dataset_key == 'grip_strength' and data.shape[1] == 1:
                    data = np.hstack([data, np.zeros_like(data)])
            if dataset_key == 'states':
                data = data[:, 1:4]
            demonstrations.append(data)
            # mid_idx = len(data) // 2
            # print(f"Loaded {filename}: Shape {data.shape}")
            # print(f"  Start: {data[:2]}")  
            # print(f"  Middle: {data[mid_idx-1:mid_idx+1]}")  
            # print(f"  End: {data[-2:]}")  
    return demonstrations


# Demonstration duration (seconds)
demo_duration = 15.0

demonstrations = load_demonstrations('../demonstration_collection/smoothed_demonstrations', dataset_key='eef_positions') # TODO: make better
gripper_demos = load_demonstrations('../demonstration_collection/smoothed_demonstrations', dataset_key='grip_strength')
# for i, demo in enumerate(demonstrations):
#     print("Demonstration {} shape: {}".format(i, demo.shape))

# Instantiate and fit the GMM_GMR model using the processed demonstrations.
gmm_gmr = GMM_GMR(demonstrations, 3, demo_duration=demo_duration)
gmm_gmr.fit()

gmm_gmr_gripper = GMM_GMR(gripper_demos, 2, demo_duration=demo_duration)
gmm_gmr_gripper.fit()

# Set up subplots for a 3-dimensional demonstration (x, y, z)
fig, axarr = plt.subplots(3, 1, figsize=(8, 12))

# Creates a time vector that spans from 0 to demo_duration seconds for each demonstration
for i in range(len(gmm_gmr.trajectories)):
    T = gmm_gmr.trajectories[i].shape[0]
    time_vec = np.linspace(0, demo_duration, T)
    for j in range(3):
        axarr[j].plot(time_vec,
                      gmm_gmr.trajectories[i, :, j],
                      linestyle=':',
                      label='Demo {}'.format(i) if i == 0 else "")

for j in range(3):
    axarr[j].scatter(gmm_gmr.centers_temporal,
                     gmm_gmr.centers_spatial[:, j],
                     label='centers')

# Generates an estimated trajectory using GMR

num_samples = 100
times, trj = gmm_gmr.generate_trajectory(0.1, num_samples)
_, gripper_trj = gmm_gmr_gripper.generate_trajectory(0.1, num_samples)

gripper_trj = gripper_trj[:, :1]

save_skill_to_h5(times, trj, gripper_trj)

for j in range(3):
    axarr[j].plot(times, trj[:, j], label='estimated')

axarr[0].set_ylabel('Y')
axarr[1].set_ylabel('X')
axarr[2].set_ylabel('Z')
axarr[2].set_xlabel('Time (s)')
axarr[0].set_title('GMM-GMR: Demonstrations & Estimated Trajectory (EEF)')

# Add legends and show the plot
for ax in axarr:
    ax.legend()

plt.savefig(os.path.join(plots_dir, 'gmm_gmr_result.png'))
plt.show()

# 3D plots
fig3d = plt.figure(figsize=(10, 8))
ax3d = fig3d.add_subplot(111, projection='3d')

for i, demo in enumerate(demonstrations):
    ax3d.plot(demo[:, 0], demo[:, 1], demo[:, 2], linestyle='--', marker='o', markersize=3,label=f'Demo {i}')

# Plot 3D trajectory
ax3d.plot(trj[:, 0], trj[:, 1], trj[:, 2],
          linewidth=2, color='red', label='Estimated Trajectory')

ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
ax3d.set_title('3D Demonstrations and Estimated Trajectory')
ax3d.legend()

plt.savefig(os.path.join(plots_dir, 'gmm_gmr_result_3d.png'))
plt.show()

# Plotting gripper strength demonstrations and the estimated trajectory
plt.figure(figsize=(8, 4))
for i in range(len(gripper_demos)):
    T = gripper_demos[i].shape[0]
    time_vec = np.linspace(0, demo_duration, T)
    plt.plot(time_vec,
             gripper_demos[i][:, 0],
             linestyle=':', 
             label='Demo {}'.format(i) if i == 0 else "")

plt.scatter(gmm_gmr_gripper.centers_temporal,
            gmm_gmr_gripper.centers_spatial[:, 0],
            label='Centers')

plt.plot(times, gripper_trj[:, 0], linewidth=2, color='red', label='Estimated Gripper Strength')
plt.xlabel('Time (s)')
plt.ylabel('Gripper Strength')
plt.title('GMM-GMR: Demonstrations & Estimated Gripper Strength')
plt.legend()

plt.savefig(os.path.join(plots_dir, 'gmm_gmr_result_gripper.png'))
plt.show()

