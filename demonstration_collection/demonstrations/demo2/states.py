import h5py
import numpy as np

# Change this path to the demonstration file you want to inspect.
file_path = "demo_1.h5"

with h5py.File(file_path, "r") as f:
    states = np.array(f["states"])
    print("States shape:", states.shape)
    print("First state (flattened):\n", states[0])
    
    # Optionally, print a few more samples
    for i in range(5):
        print(f"State {i}:", states[i])
