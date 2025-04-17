# GMM/GMR Learning from Demonstration Implementation

This repository contains an implementation of the **Learning from Demonstration (LfD)** technique using **Gaussian Mixture Models (GMM)** and **Gaussian Mixture Regression (GMR)**. This project was completed for **COMP.5495 - Robot Learning**.

The implementation is inspired by the code available at [ceteke/GMM-GMR](https://github.com/ceteke/GMM-GMR), which is based on the following paper:

> **Calinon, S., Guenter, F., & Billard, A. (2007).**  
> *"On Learning, Representing, and Generalizing a Task in a Humanoid Robot."*  
> IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics).  
> [Available here](https://ieeexplore.ieee.org/document/4126276/).

This adaptation follows the same licensing terms as the original repository.

---

## Environment Setup

This project is developed using the **robosuite** framework with the **UR5e** robot on **Ubuntu 22.04/24.04** in an **Anaconda** environment. Below are the steps to set up the environment and run the code.

### Prerequisites

- Ubuntu 22.04 or 24.04
- Anaconda
- robosuite
- Controller for collecting demonstrations (currently not compatible with keyboard)

### Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/JohnBrann/gmm-gmr
   cd gmm-gmr

2. **Create Conda Environment**
   ```bash
   conda env create -f robosuite_env.yml
---

### Running Code
1. **Activate Conda**
   ```bash
   conda activate robosuite_env
2. **Run Bash Script**  
   ```bash
   ./pipeline_script.sh
   
Note: Before running, make sure there are no demonstrations, smoothed_demonstrations, or skill files directly in their associated folders

---

### Demonstration Controls
Below are the controls to be used when providing demonstrations. Controller layout is Xbox but the program should also work with PlayStation controllers in the way you'd expect.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b6d40da0-0977-4834-99cf-3e9248e3d472" width="360"/>
  <!--<p align="center">
    <em>Could put a caption here, probably don't need to tho</em>
  </p>-->
</p>

---

## Results
Below are some images showing demonstrated trajectories and gmm/gmr being applied to those trajectories:
