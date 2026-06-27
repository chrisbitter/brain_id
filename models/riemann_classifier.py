"""
Implementation based on these papers:
1) "Fixed Point Algorithms for Estimating Power Means of Positive Definite Matrices" by Marco Congedo, Alexandre Barachant, and Ehsan Kharati Koopaei (2017)
2) "Riemannian geometry for EEG-based brain-computer interfaces; a primer and a review" by M Congedo and A Barachant and R Bhatia (2017)
"""

import numpy as np
from scipy.linalg import sqrtm, inv
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA



def regularized_covariance(data: np.ndarray) -> np.ndarray:
    """Data shape: (n_samples, n_features)"""
    return LedoitWolf().fit(data).covariance_


def is_spd(matrix: np.ndarray) -> bool:
    """Check if a matrix is real symmetric positive definite (SPD)."""
    is_real = not np.iscomplexobj(matrix)
    is_symm = np.allclose(matrix, matrix.T, atol=1e-5)
    is_posd = np.all(np.linalg.eigh(matrix)[0] > 0)
    # if not is_real: print("Matrix must be real")  # Debug print
    # if not is_symm: print("Matrix must be symmetric")  # Debug print
    # if not is_posd: print("Matrix must be positive definite")  # Debug print
    return is_real and is_symm and is_posd


def matrix_power(M: np.ndarray, p: float, eps: float=1e-12) -> np.ndarray:
    # Eigen decomposition (guaranteed real for SPD matrices)
    eigvals, eigvecs = np.linalg.eigh(M)
    # Regularize eigenvalues to prevent negatives/near-zero
    eigvals_clamped = np.maximum(eigvals, eps)
    # Compute power and reconstruct matrix
    powered = eigvecs @ np.diag(eigvals_clamped ** p) @ eigvecs.T
    # Ensure symmetry (already guaranteed, but safe)
    powered = 0.5 * (powered + powered.T)
    return powered


def sym_inv(M: np.ndarray) -> np.ndarray:
    """Compute matrix inverse and ensure symmetry."""
    M_inv = inv(M)
    M_inv = (M_inv + M_inv.T) * 0.5
    return M_inv


def ensure_spd(M: np.ndarray, reg: float = 1e-12) -> np.ndarray:
    """
    Ensure matrix is SPD by adding a small regularization term.
    
    Parameters:
    M (np.ndarray): Input matrix
    reg (float): Regularization parameter
    
    Returns:
    np.ndarray: Regularized SPD matrix
    """
    M = matrix_power(M, 1, reg)  # Ensure positive definiteness
    M = 0.5 * (M + M.T)  # Ensure symmetry
    return M


def power_mean(covariances, p, weights=None, tol=1e-12, max_iter=100, eps=1e-6, reg=1e-10):
    """
    Compute power mean of SPD matrices using MPM algorithm.
    
    Parameters:
    covariances (list): List of SPD matrices (N x N)
    p (float): Power parameter in (-1, 1)\{0}
    weights (np.ndarray): Optional weights (default: uniform)
    tol (float): Convergence tolerance
    max_iter (int): Maximum iterations
    eps (float): Small value to check for small eigenvalues
    reg (float): matrix regularization parameter
    
    Returns:
    np.ndarray: Power mean matrix
    """
    
    # approach requires double precision
    # susceptible to numerical errors when using single precision
    dtype = np.float64
    covariances = [C.astype(dtype) for C in covariances]

    assert p != 0, "Power parameter cannot be zero"
    assert -1 < p < 1, "Power parameter must be in (-1, 1)"
    if weights is not None:
        assert np.sum(weights) == 1, "Weights must sum to 1"
    for mat in covariances:
        assert is_spd(mat), "Input matrices must be SPD"
        assert np.all(np.isfinite(mat)), "Input matrices must be finite"
        assert not np.iscomplexobj(mat), "Input matrices must be real"

    # check for small eigenvalues
    eigv = np.array([np.linalg.eigh(C)[0] for C in covariances])
    conds = np.max(eigv, axis=1) / np.min(eigv, axis=1)
    if np.any(conds > 1 / eps):
        print("WARNING: Small eigenvalues detected in input matrices. Applying regularization...")
        covariances = [C + reg * np.eye(C.shape[0]) for C in covariances]

    if p < 0:
        # regularized inverse
        covariances = [sym_inv(C) for C in covariances]
        assert all(is_spd(C) for C in covariances), "inverted matrices must be real SPD"
    
    K = len(covariances)
    N = covariances[0].shape[0]
    weights = np.ones(K, dtype=dtype)/K if weights is None else weights
    
    # Initialization using commuting case solution
    p_abs = abs(p)
    powered_covs = [matrix_power(C, p_abs) for C in covariances]
    G_initial = sum(w * C for w, C in zip(weights, powered_covs))
    G_initial = matrix_power(G_initial, 1/p_abs)
    assert is_spd(G_initial), "G must be real SPD"
    
    # Initialize X as inverse sqrt of G_initial
    X = matrix_power(G_initial, 0.5) if p > 0 else sym_inv(sqrtm(G_initial))
    assert is_spd(X), "X must be real SPD"
    
    # Heuristic phi value from paper, phi must be in [0.25, 0.5]
    phi = 0.375 / p_abs
    
    for _ in range(max_iter):
        # Compute H
        H = np.zeros((N, N), dtype=dtype)
        for C, w in zip(covariances, weights):
            term = X @ C @ X.T
            H += w * matrix_power(term, p_abs)
            assert is_spd(H), "H must be real SPD"
        
        # Check convergence
        conv = np.linalg.norm(H - np.eye(N)) / np.sqrt(N)
        # print(f"Iteration {_+1}, convergence: {conv:.2e}")
        if conv < tol:
            break
        
        # Update X using H^{-phi}
        H_power = matrix_power(H, -phi)
        X = H_power @ X
        X = ensure_spd(X)
    
    if conv >= tol:
        print(f"WARNING: Power mean did not converge after {max_iter} iterations.")
        print(f"Convergence value: {conv:.2e}, tolerance: {tol:.2e}")

    # Reconstruct P from X
    X = ensure_spd(X)
    P = sym_inv(X.T @ X) if p > 0 else X.T @ X
    P = ensure_spd(P)
    return P


def riemann_geodesic(C1, C2, t=0.5):
    """Compute a point along the geodesic path at t between two SPD matrices. """
    C1_sqrt = matrix_power(C1, 0.5)
    C1_inv_sqrt = sym_inv(C1_sqrt)
    M = C1_inv_sqrt @ C2 @ C1_inv_sqrt
    M = matrix_power(M, t)
    M = C1_sqrt @ M @ C1_sqrt
    M = ensure_spd(M)
    return M


def riemann_mean(covariances, weights=None, tol=1e-8, max_iter=100, eps=1e-6, reg=1e-12):
    """
    Compute Riemannian geometric mean using midpoint of p=±0.01 power means.
    
    Parameters:
    covariances (list): List of SPD matrices (N x N)
    weights (np.ndarray): Optional weights
    tol (float): Convergence tolerance
    max_iter (int): Maximum iterations
    eps (float): Small value to check for small eigenvalues
    reg (float): matrix regularization parameter

    Returns:
    np.ndarray: Estimated geometric mean matrix
    """
    # Compute power means at p=±0.01
    covariances = [C.astype(np.float64) for C in covariances]
    G_p = power_mean(covariances, 0.01, weights, tol, max_iter, eps, reg)
    G_n = power_mean(covariances, -0.01, weights, tol, max_iter, eps, reg)
    
    # Compute geodesic midpoint (G_p #_{0.5} G_n)
    midpoint = riemann_geodesic(G_p, G_n, t=0.5)
    return midpoint


def riemann_distance(C1, C2, reg=1e-12):
    """Compute Riemannian distance between two SPD matrices"""

    C1 = C1.astype(np.float64)
    C2 = C2.astype(np.float64)

    # Check if the input matrices are SPD
    if not is_spd(C1):
        print("WARNING: C1 is not SPD. Applying regularization...")
        C1 = ensure_spd(C1, reg)
    if not is_spd(C2):
        print("WARNING: C2 is not SPD. Applying regularization...")
        C2 = ensure_spd(C2, reg)

    # Compute similarity transformation
    inv_sqrt_C1 = sym_inv(matrix_power(C1, 0.5))
    M = inv_sqrt_C1 @ C2 @ inv_sqrt_C1
    M = ensure_spd(M, reg)
    
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, reg)
    dist = np.sqrt(np.sum(np.log(eigvals)**2))
    return dist


class RiemannMDM:
        
        """
        RiemanMDM: A Minimum Distance to Mean classifier based on the Riemannian manifold of SPD matricies.
        This class implements a classifier that operates on covariance matrices derived 
        from multi-channel time-series data. It uses Riemannian geometry to compute 
        class means and classify new samples based on their Riemannian distance to these means.
        """

        def __init__(self, use_pca:bool=True, pca_components:int=0.95):
            """ Initializes an instance of the class. """
            self.use_pca = use_pca
            self.pca_components = pca_components
            self.classes = None
            self.means = None
            self.pca = None
            self.n_components = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> object:
            """
            Fits the Riemannian means of covariance matrices for each class.
            This method computes the covariance over the channel dimension.

            Parameters:
                X (array-like of shape (n_samples, n_channels, n_timesteps), dtype float):
                    The input data, where each sample is a multi-channel time-series data matrix.
                y (array-like of shape (n_samples,), dtype int):
                    The class labels corresponding to each sample in X.

            Returns:
                self : object
                Returns the instance of the class with the computed Riemannian means stored.
            """

            if self.use_pca:
                n_samples, n_channels, n_timesteps = X.shape
                self.pca = PCA(n_components=self.pca_components, whiten=True)
                X = np.transpose(X, (0, 2, 1))  # (n_samples, n_timesteps, n_channels)
                X = X.reshape(-1, X.shape[2])  # (n_samples * n_timesteps, n_channels)
                X = self.pca.fit_transform(X)
                X = X.reshape(n_samples, n_timesteps, X.shape[1])  # (n_samples, n_timesteps, n_components)
                X = np.transpose(X, (0, 2, 1))  # (n_samples, n_components, n_timesteps)
                self.n_components = X.shape[1]
                print(f"Reduced to {self.n_components} components using PCA")

            self.classes = np.unique(y)
            self.means = {}

            for c in self.classes:
                Xc = X[y == c]
                Xc = [regularized_covariance(x.T) for x in Xc]
                # try:
                self.means[c] = riemann_mean(Xc, reg=1e-6)
                # except Exception as e:
                #     print(f"Error computing mean for class {c}: {e}")
                #     self.means[c] = np.eye(X.shape[1])
            return self
        
        def predict(self, X: list) -> tuple:
            """
            Predict the class of time-series signals.

            This method takes a list of time-series signal matrices, computes their covariance,
            and predicts the class for each signal based on the Riemannian distance 
            to the class means. A numerically stable softmax is applied to the distances 
            for a normalized similarity distribution.

            Parameters:
                X (array-like of shape (batch, n_channels, n_timesteps)): 
                    A list of EEG signal matrices, where each matrix represents a single trial.

            Returns:
                tuple (predicted_labels, distance_probs):
                    - ndarray: An array of predicted class labels. shape (batch,)
                    - ndarray: An array of distance-based probabilities for each class. shape (batch, n_classes)
            """
            y_pred = np.empty(len(X), dtype=int)

            if self.pca is not None:
                # apply pca on the channel dimension (n_channels -> n_components) 
                batch, n_channels, n_timesteps = X.shape
                X = np.transpose(X, (0, 2, 1))
                X = X.reshape(-1, X.shape[2])
                X = self.pca.transform(X)
                X = X.reshape(batch, n_timesteps, X.shape[1])
                X = np.transpose(X, (0, 2, 1))

            for i, x in enumerate(X):
                x = regularized_covariance(x.T)
                # since cov lays on a hypercone in the SPD manifold matrices with small values have a bad tangent space approximation
                # so we add a small regularization term to the covariance matrix
                dists = [riemann_distance(x, self.means[c], reg=1e-9) for c in self.classes]
                dists = np.array(dists)

                y_pred[i] = self.classes[np.argmin(dists)]

                # convert distances to "probabilities" numerically stable softmax
                prob = np.exp(-dists - np.max(-dists))
                prob = prob / np.sum(prob)

            return np.array(y_pred), prob
