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

def save_skill_to_h5(times, trajectory, attrs, folder_path="skills"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Determine the next available file number in the folder
    existing_files = os.listdir(folder_path)
    pattern = f"skill_{attrs['skill_name']}_" + r"(\d+)\.h5"
    max_num = 0
    for filename in existing_files:
        m = re.match(pattern, filename)
        if m:
            num = int(m.group(1))
            if num > max_num:
                max_num = num
    new_file_num = max_num + 1
    new_file_name = f"skill_{attrs['skill_name']}_{new_file_num}.h5"
    full_path = os.path.join(folder_path, new_file_name)
    
    # Save data
    with h5py.File(full_path, "w") as f:
        f.attrs.update(attrs)
        f.create_dataset("times", data=times)
        f.create_dataset("trajectory", data=trajectory)
    
    print(f"Skill saved to {full_path}")

# Loads demonstrations in h5 format
def load_demonstrations(folder_path, dataset_key='eef_positions'):
    skill_demos = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.h5'):
            filepath = os.path.join(folder_path, filename)
            attrs = {}
            skill_name = ""
            with h5py.File(filepath, 'r') as f:
                data = np.array(f[dataset_key])
                attrs.update(f.attrs)
                if "skill_name" not in attrs.keys():
                    print(f"Skipping demos from {filepath} due to missing attributes!")
                    continue
                skill_name = attrs["skill_name"]
                if data.ndim < 2:
                    data = data.reshape(-1, 1)
            if dataset_key == 'states':
                data = data[:, 1:4]
            if skill_name in skill_demos.keys():
                skill_demos[skill_name]["demos"].append(data)
            else:
                skill_demos[skill_name] = { "attrs": attrs, "demos": [data] }
            # mid_idx = len(data) // 2
            # print(f"Loaded {filename}: Shape {data.shape}")
            # print(f"  Start: {data[:2]}")  
            # print(f"  Middle: {data[mid_idx-1:mid_idx+1]}")  
            # print(f"  End: {data[-2:]}")  
    return skill_demos


def learn_skill(skill_demos):
    demonstrations = skill_demos["demos"]
    attrs          = skill_demos["attrs"]

    # Fit GMM-GMR
    gmm_gmr = GMM_GMR(demonstrations, 3, demo_duration=demo_duration)
    gmm_gmr.fit()

    # Prepare 2D subplot
    fig, axarr = plt.subplots(3, 1, figsize=(8, 12))
    num_samples = 100

    # Plot each demo, remapped to 0…num_samples-1
    for i, traj in enumerate(gmm_gmr.trajectories):
        T        = traj.shape[0]
        demo_t   = np.linspace(0, demo_duration, T)
        demo_idx = demo_t / demo_duration * (num_samples - 1)

        for j in range(3):
            axarr[j].plot(
                demo_idx,
                traj[:, j],
                linestyle=':',
                label='Demo {}'.format(i) if i == 0 else ""
            )

    # Plot Gaussian centers at their sample‐indices
    center_idx = (gmm_gmr.centers_temporal / demo_duration * (num_samples - 1)).astype(int)
    for j in range(3):
        axarr[j].scatter(
            center_idx,
            gmm_gmr.centers_spatial[:, j],
            s=50,
            label='centers'
        )

    # Generate & save the estimated trajectory
    times, trj = gmm_gmr.generate_trajectory(0.1, num_samples)
    save_skill_to_h5(times, trj, attrs)

    # Plot the estimate on the same 0 - 99 axis
    est_idx = np.arange(num_samples)
    for j in range(3):
        axarr[j].plot(
            est_idx,
            trj[:, j],
            linewidth=2,
            label='estimated'
        )

    # 7) Finalize labels, legend, title
    axarr[0].set_ylabel('Y')
    axarr[1].set_ylabel('X')
    axarr[2].set_ylabel('Z')
    axarr[2].set_xlabel('Sample index')
    axarr[0].set_title(
        f"GMM-GMR: Demonstrations & Estimated Trajectory (EEF) for '{attrs['skill_name']}'"
    )
    for ax in axarr:
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"gmm_gmr_{attrs['skill_name']}_result.png"))
    plt.show()

    # 8) (Optional) 3D plot remains unchanged:
    fig3d = plt.figure(figsize=(10, 8))
    ax3d  = fig3d.add_subplot(111, projection='3d')
    for i, demo in enumerate(demonstrations):
        ax3d.plot(
            demo[:, 0], demo[:, 1], demo[:, 2],
            linestyle='--',
            marker='o',
            markersize=3,
            label=f'Demo {i}'
        )
    ax3d.plot(
        trj[:, 0], trj[:, 1], trj[:, 2],
        linewidth=2,
        color='red',
        label='Estimated Trajectory'
    )
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title(f"3D Demonstrations & Estimated Trajectory for '{attrs['skill_name']}'")
    ax3d.legend()
    plt.savefig(os.path.join(plots_dir, f"gmm_gmr_{attrs['skill_name']}_result_3d.png"))
    plt.show()



# Demonstration duration (seconds)
demo_duration = 100.0

sk_demos = load_demonstrations('../demonstration_collection/smoothed_demonstrations', dataset_key='eef_positions') # TODO: make better

for demos in sk_demos.values():
    learn_skill(demos)
