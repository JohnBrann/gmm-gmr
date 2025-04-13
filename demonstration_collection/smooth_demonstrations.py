import os
import h5py
import numpy as np
from scipy.interpolate import interp1d

# Smooth the trajectory using cubic interpolation.
# def smooth_trajectory(data, num_samples=200, kind='cubic'):
def smooth_trajectory(data, num_samples=50, kind='cubic'):
    n_timesteps, n_features = data.shape if data.ndim > 1 else (data.shape[0], 1)
    # Create a normalized time vector for the raw data.
    original_t = np.linspace(0, 1, n_timesteps)
    # Create a new time vector for the smoothed data.
    resampled_t = np.linspace(0, 1, num_samples)
    smoothed = np.zeros((num_samples, n_features))
    
    # Interpolate each feature separately.
    if n_features > 1:
        for i in range(n_features):
            f_interp = interp1d(original_t, data[:, i], kind=kind, fill_value="extrapolate")
            smoothed[:, i] = f_interp(resampled_t)
    else:
        f_interp = interp1d(original_t, data, kind=kind, fill_value="extrapolate")
        smoothed = f_interp(resampled_t)
    
    return smoothed

raw_demo_folder = "demonstrations"
smoothed_demo_folder = "smoothed_demonstrations"

if not os.path.exists(smoothed_demo_folder):
    os.makedirs(smoothed_demo_folder)
files = [f for f in os.listdir(raw_demo_folder) if f.endswith('.h5')]

num_samples = 50 # Number of samples on curve

for filename in files:
    raw_file_path = os.path.join(raw_demo_folder, filename)
    
    # Open the raw demonstration file.
    with h5py.File(raw_file_path, "r") as f:
        raw_timestamps = np.array(f["timestamps"])
        raw_positions = np.array(f["eef_positions"])
        raw_grip_strength = np.array(f["grip_strength"])

    # Apply smoothing
    smoothed_positions = smooth_trajectory(raw_positions, num_samples=num_samples, kind='cubic')
    smoothed_grip_strength = smooth_trajectory(raw_grip_strength, num_samples=num_samples, kind='cubic')
    # Interpolate timestamps to match the new number of samples.
    smoothed_timestamps = np.linspace(raw_timestamps[0], raw_timestamps[-1], num_samples)
    
    new_filename = "smoothed_" + filename
    smoothed_file_path = os.path.join(smoothed_demo_folder, new_filename)
    
    with h5py.File(smoothed_file_path, "w") as f:
        f.create_dataset("timestamps", data=smoothed_timestamps)
        f.create_dataset("grip_strength", data=smoothed_grip_strength)
        f.create_dataset("eef_positions", data=smoothed_positions)
    
    print(f"Saved smoothed demonstration: {smoothed_file_path}")

