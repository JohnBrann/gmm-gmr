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

import numpy as np
from utils import align_trajectories, gaussian
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

class GMM_GMR(object):
    """
    Implementation of GMM-GMR based imitation.
    
    This version has been modified to scale the temporal dimension so that the generated
    trajectory reflects the actual demonstration time (e.g., 5 seconds) instead of using
    raw step indices.
    """
    def __init__(self, trajectories, n_components, demo_duration=5.0):
        """
        :param trajectories: Trajectories obtained from demonstrations. If these are not
            an aligned numpy array (i.e. list of non-aligned trajectories) they are aligned.
            (Each trajectory is assumed to have shape (T, D), where D includes spatial data.)
        :param n_components: Number of PCA components.
        :param demo_duration: The actual duration (in seconds) of the demonstration.
            This is used to scale the temporal dimension.
        """
        self.demo_duration = demo_duration
        
        if isinstance(trajectories, list):
            self.trajectories = np.array(align_trajectories(trajectories))
        else:
            self.trajectories = trajectories

        self.T = self.trajectories.shape[1]  # number of time steps
        self.N = self.trajectories.shape[0]  # number of demonstrations
        self.D = self.trajectories.shape[2]  # data dimensions

        self.pca = PCA(n_components)

    def fit(self):
        # Flatten the trajectories for PCA
        trajectories_latent = self.pca.fit_transform(self.trajectories.reshape(-1, self.D))
        print("Explained variance: {}%".format(np.sum(self.pca.explained_variance_ratio_) * 100))

        # Scale the time axis so that the entire demonstration lasts demo_duration seconds.
        time_scale = self.demo_duration / self.T
        # Instead of using raw indices 0,1,...,T-1, we use scaled time in seconds.
        temporal = np.array([np.arange(self.T) * time_scale] * self.N).reshape(-1, 1)

        spatio_temporal = np.concatenate((temporal, trajectories_latent), axis=1)

        # Use BIC to select the best number of mixtures
        # components = [5, 10, 15, 20, 25, 30, 35, 40, 45]
        components = [2,3,4,5] # number of gaussians 
        bics = []
        for c in components:
            gmm = GaussianMixture(n_components=c)
            gmm.fit(spatio_temporal)
            bics.append(gmm.bic(spatio_temporal))

        c = components[np.argmin(bics)]
        print("Selected n mixtures: {}".format(c))

        self.gmm = GaussianMixture(n_components=c)
        self.gmm.fit(spatio_temporal)
        print("Is GMM converged: ", self.gmm.converged_)

        self.gmr = GMR(self.gmm)
        self.centers = self.gmm.means_
        self.centers_temporal = self.centers[:, 0]  # These are now in seconds
        self.centers_spatial_latent = self.centers[:, 1:]
        self.centers_spatial = self.pca.inverse_transform(self.centers_spatial_latent)

    def generate_trajectory(self, interval=0.1):
        """
        Generate a trajectory using GMR.
        
        :param interval: The sampling interval (in seconds) for the generated trajectory.
        :return: A tuple (times, trajectory), where 'times' are in seconds and 'trajectory'
                 is the spatial data reconstructed from the latent space.
        """
        times = np.arange(min(self.centers_temporal), max(self.centers_temporal) + interval, interval)
        trj = []
        for t in times:
            trj.append(self.gmr.estimate(t))
        trj = np.squeeze(np.array(trj))
        trj = self.pca.inverse_transform(trj)
        return times, trj

class GMR:
    def __init__(self, gmm):
        self.gmm = gmm
        self.n_components = self.gmm.means_.shape[0]

    def xi_s_k(self, xi_t, k):
        cov_k = self.gmm.covariances_[k]
        mu_k = self.gmm.means_[k]

        mu_t_k = mu_k[0:1].reshape(-1, 1)
        mu_s_k = mu_k[1:].reshape(-1, 1)
        cov_t_k = cov_k[0:1, 0:1]
        cov_st_k = cov_k[1:, 0:1]

        return mu_s_k + cov_st_k.dot(np.linalg.inv(cov_t_k)).dot(xi_t - mu_t_k)

    def get_denom(self, xi_t):
        probs = 0.0
        for k in range(self.n_components):
            mu_t_k = self.gmm.means_[k][0]
            var_t_k = self.gmm.covariances_[k][0, 0]
            probs += gaussian(xi_t, mu_t_k, var_t_k)
        return probs

    def estimate(self, xi_t):
        result = 0.0
        for k in range(self.n_components):
            xi_s_k_head = self.xi_s_k(xi_t, k)
            mu_t_k = self.gmm.means_[k][0]
            var_t_k = self.gmm.covariances_[k][0, 0]
            beta_k = gaussian(xi_t, mu_t_k, var_t_k) / self.get_denom(xi_t)
            result += xi_s_k_head * beta_k
        return result
