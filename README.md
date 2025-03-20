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

This project is developed using the **robosuite** framework with the **UR5e** robot on **Ubuntu 22.04** in an **Anaconda** environment. Below are the steps to set up the environment and run the code.

### Prerequisites

- Ubuntu 22.04
- Anaconda
- Python 3.x
- robosuite

### Installation

1. **Clone the repository**  
   ```bash
   git clone <repo-url>
   cd <repo-name>

2. **Create Conda Environment**
   ```bash
   conda env create -f robosuite_env.yml
---

## Results
Below are some images showing demonstrated trajectories and gmm/gmr being applied to those trajectories:
