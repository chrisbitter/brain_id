import numpy as np
from sklearn.decomposition import PCA

from features import channel_covariance, power_spectra


class FeatureExtractor:

    """ Extract features from the plain EEG data.
    This is especially useful for ML algorithms to reduce the dimensionality of the data.
    """

    def __init__(self, *args, **kwargs):
        self.extractor = ReducedChannelCovariance(*args, **kwargs)

    def fit(self, X):
        """ Fit the feature extractor to the data.
        Args:
            X (np.ndarray): sample data (n_samples, n_channels, n_times)
        """
        self.extractor.fit(X)

    def transform(self, X):
        """ Transform the data using the fitted feature extractor.
        Args:
            X (np.ndarray): sample data (n_samples, n_channels, n_times)
        Returns:
            np.ndarray: transformed data (n_samples, n_features)
        """
        
        # TODO selecting relevant features is important. 
        # Try experimenting with the provided methods or 
        # try different methods you find in the literature.

        return self.extractor.transform(X)

    def fit_transform(self, X):
        """ Fit the feature extractor to the data and transform it. """
        self.fit(X)
        return self.transform(X)
    

class SimplePCA:

    """ Extract PCA components from the flattened EEG window. """

    def __init__(self, pca_components=0.95):
        self.pca_components = pca_components
        self.pca = None

    def fit(self, X):
        """ Fit the PCA to the eeg window. """

        # flatten eeg window
        n_samples, n_channels, n_times = X.shape
        X = X.reshape(n_samples, -1)

        # fit PCA
        self.pca = PCA(n_components=self.pca_components, whiten=True)
        self.pca.fit(X)

        # print(f"Reduced from {n_channels * n_times} to {self.pca.n_components_} components using PCA")

    def transform(self, X):
        """ Transform the data using the fitted PCA. """
        
        # reduce dimensionality
        X = X.reshape(X.shape[0], -1)  # flatten eeg window
        X_red = self.pca.transform(X)

        return X_red


class ReducedChannelCovariance:

    """ Reduce the number of channels using PCA and compute the covariance matrix over channels. """

    def __init__(self, pca_components=0.95):
        self.pca_components = pca_components

    def fit(self, X):

        n_samples, n_channels, n_timesteps = X.shape
        self.pca = PCA(n_components=self.pca_components, whiten=True)
        X = np.transpose(X, (0, 2, 1))                     # (n_samples, n_timesteps, n_channels)
        X = X.reshape(-1, X.shape[2])                      # (n_samples * n_timesteps, n_channels)
        X = self.pca.fit_transform(X)
        X = X.reshape(n_samples, n_timesteps, X.shape[1])  # (n_samples, n_timesteps, n_components)
        X = np.transpose(X, (0, 2, 1))                     # (n_samples, n_components, n_timesteps)
        self.n_components = X.shape[1]
        # print(f"Reduced to {self.n_components} components using PCA")

    def transform(self, X):
        batch, n_channels, n_timesteps = X.shape
        
        # apply PCA on the channel dimension
        X = np.transpose(X, (0, 2, 1))
        X = X.reshape(-1, X.shape[2])
        X = self.pca.transform(X)
        X = X.reshape(batch, n_timesteps, X.shape[1])
        X = np.transpose(X, (0, 2, 1))

        # compute covariance matrix
        X_cov = channel_covariance(X)

        # return flattened upper triangular part of the covariance matrix
        idxs = np.triu_indices(X_cov.shape[-1], k=1)
        X_cov = X_cov[:, idxs[0], idxs[1]]
        X_cov = X_cov.reshape(X.shape[0], -1)

        return X_cov
