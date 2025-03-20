import os
import h5py
import numpy as np
from scipy.interpolate import interp1d

# Define a function to smooth the trajectory using cubic interpolation.
def smooth_trajectory(data, num_samples=200, kind='cubic'):
    """
    Smooth the given data by interpolating each dimension separately.
    
    Parameters:
        data (np.ndarray): Raw data with shape (n_timesteps, n_features).
        num_samples (int): Number of samples for the smoothed trajectory.
        kind (str): Type of interpolation ('cubic', 'linear', etc.).
        
    Returns:
        np.ndarray: Smoothed data with shape (num_samples, n_features).
    """
    n_timesteps, n_features = data.shape
    # Create a normalized time vector for the raw data.
    original_t = np.linspace(0, 1, n_timesteps)
    # Create a new time vector for the smoothed data.
    resampled_t = np.linspace(0, 1, num_samples)
    smoothed = np.zeros((num_samples, n_features))
    
    # Interpolate each feature separately.
    for i in range(n_features):
        f_interp = interp1d(original_t, data[:, i], kind=kind, fill_value="extrapolate")
        smoothed[:, i] = f_interp(resampled_t)
    
    return smoothed

# Folder paths
raw_folder = "demonstrations"
smoothed_folder = "smoothed_demonstrations"

# Create the smoothed demonstrations folder if it doesn't exist.
if not os.path.exists(smoothed_folder):
    os.makedirs(smoothed_folder)

# Get a list of all HDF5 files in the demonstrations folder.
files = [f for f in os.listdir(raw_folder) if f.endswith('.h5')]

# User-defined number of samples for the smoothed trajectory.
num_samples = 200

for filename in files:
    raw_file_path = os.path.join(raw_folder, filename)
    
    # Open the raw demonstration file.
    with h5py.File(raw_file_path, "r") as f:
        # Load the raw timestamps and end-effector positions.
        raw_timestamps = np.array(f["timestamps"])
        raw_positions = np.array(f["eef_positions"])  # Expected shape: (n_timesteps, 3)
    
    # Apply smoothing to the end-effector positions.
    smoothed_positions = smooth_trajectory(raw_positions, num_samples=num_samples, kind='cubic')
    # Interpolate timestamps to match the new number of samples.
    smoothed_timestamps = np.linspace(raw_timestamps[0], raw_timestamps[-1], num_samples)
    
    # Create a new filename and file path for the smoothed demonstration.
    new_filename = "smoothed_" + filename
    smoothed_file_path = os.path.join(smoothed_folder, new_filename)
    
    # Save the smoothed data to the new HDF5 file.
    with h5py.File(smoothed_file_path, "w") as f:
        f.create_dataset("timestamps", data=smoothed_timestamps)
        f.create_dataset("eef_positions", data=smoothed_positions)
        # If you wish to smooth and save actions, add similar code here.
    
    print(f"Saved smoothed demonstration: {smoothed_file_path}")



# import os
# import h5py
# import numpy as np
# from scipy.interpolate import interp1d

# # Define a function to smooth the trajectory using cubic interpolation.
# def smooth_trajectory(data, num_samples=200, kind='cubic'):
#     """
#     Smooth the given data by interpolating each dimension separately.
    
#     Parameters:
#         data (np.ndarray): Raw data with shape (n_timesteps, n_features).
#         num_samples (int): Number of samples for the smoothed trajectory.
#         kind (str): Type of interpolation ('cubic', 'linear', etc.).
        
#     Returns:
#         np.ndarray: Smoothed data with shape (num_samples, n_features).
#     """
#     n_timesteps, n_features = data.shape
#     # Create a normalized time vector for the raw data.
#     original_t = np.linspace(0, 1, n_timesteps)
#     # Create a new time vector for the smoothed data.
#     resampled_t = np.linspace(0, 1, num_samples)
#     smoothed = np.zeros((num_samples, n_features))
    
#     # Interpolate each feature separately.
#     for i in range(n_features):
#         f_interp = interp1d(original_t, data[:, i], kind=kind, fill_value="extrapolate")
#         smoothed[:, i] = f_interp(resampled_t)
    
#     return smoothed

# # Folder paths
# raw_folder = "demonstrations"
# smoothed_folder = "smoothed_demonstrations"

# # Create the smoothed demonstrations folder if it doesn't exist.
# if not os.path.exists(smoothed_folder):
#     os.makedirs(smoothed_folder)

# # Get a list of all HDF5 files in the demonstrations folder.
# files = [f for f in os.listdir(raw_folder) if f.endswith('.h5')]

# # User-defined number of samples for the smoothed trajectory.
# num_samples = 200

# for filename in files:
#     raw_file_path = os.path.join(raw_folder, filename)
    
#     # Open the raw demonstration file.
#     with h5py.File(raw_file_path, "r") as f:
#         # Load the raw timestamps and states.
#         raw_timestamps = np.array(f["timestamps"])
#         raw_states = np.array(f["states"])  # Expected shape: (n_timesteps, state_dim)
    
#     # Apply smoothing to the states.
#     smoothed_states = smooth_trajectory(raw_states, num_samples=num_samples, kind='cubic')
#     # Optionally, interpolate timestamps to match the new number of samples.
#     smoothed_timestamps = np.linspace(raw_timestamps[0], raw_timestamps[-1], num_samples)
    
#     # Create a new filename and file path for the smoothed demonstration.
#     new_filename = "smoothed_" + filename
#     smoothed_file_path = os.path.join(smoothed_folder, new_filename)
    
#     # Save the smoothed data to the new HDF5 file.
#     with h5py.File(smoothed_file_path, "w") as f:
#         f.create_dataset("timestamps", data=smoothed_timestamps)
#         f.create_dataset("states", data=smoothed_states)
#         # If you wish to smooth and save actions, add similar code here.
    
#     print(f"Saved smoothed demonstration: {smoothed_file_path}")
