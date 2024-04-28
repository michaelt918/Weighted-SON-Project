from scipy.spatial import KDTree
from scipy.sparse import lil_array
from scipy.sparse import csr_array
from scipy.sparse import find
from scipy.sparse import triu
import scipy.linalg as la
import numpy as np
import numpy.typing as npt
from typing import Any
import time


def compute_weights(
    X: npt.ArrayLike, gamma: float, omega=None, verbose=1
) -> (npt.ArrayLike, csr_array, csr_array):
    if verbose:
        print("Computing weights...")
    start_time = time.perf_counter()
    N = X.shape[1]
    weight_matrix, _ = compute_weight_matrix(X, gamma, omega, N)
    idx_r, idx_c, val = find(triu(weight_matrix))
    num_weight = len(val)
    W = lil_array((N, num_weight))
    Wbar = lil_array((N, num_weight))
    # Set 1 at specified positions
    W[idx_r, np.arange(num_weight)] = 1
    Wbar[idx_c, np.arange(num_weight)] = 1
    W = W.tocsr()
    Wbar = Wbar.tocsr()
    weight_vec = val.T
    node_arc_matrix = W - Wbar
    end_time = time.perf_counter()
    if verbose:
        print("Weights computed in {} seconds.".format(end_time - start_time))
    return weight_vec, node_arc_matrix, weight_matrix, end_time - start_time


def compute_weight_matrix(X, gamma, omega, verbose=1):
    N = X.shape[1]
    if verbose:
        print("Computing weight matrix...")
    start_time = time.perf_counter()
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    weight_matrix = lil_array((N, N))

    if omega is not None:
        tree = KDTree(X.T)
        omega_balls = tree.query_ball_tree(tree, r=omega)
        for i in range(N):
            for j in omega_balls[i]:
                if j > i:
                    weight_matrix[i, j] = np.exp(-gamma * la.norm(X[:, i] - X[:, j])) * gamma**(X.shape[0]+1)
                    weight_matrix[j, i] = weight_matrix[i, j]
    else:
        for i in range(N):
            for j in range(i+1, N):
                weight_matrix[i, j] = np.exp(-gamma * la.norm(X[:, i] - X[:, j])) * gamma**(X.shape[0]+1)
                weight_matrix[j, i] = weight_matrix[i, j]
    
    weight_matrix.setdiag(0)
    weight_matrix = weight_matrix.tocsr()
    end_time = time.perf_counter()
    if verbose:
        print("Weight matrix computed in {} seconds.".format(end_time - start_time))
    return weight_matrix, end_time - start_time


def get_nam_wv_from_wm(weight_matrix: npt.ArrayLike) -> (csr_array, npt.ArrayLike):
    N = weight_matrix.shape[0]
    idx_r, idx_c, val = find(triu(weight_matrix))
    num_weight = len(val)
    W = lil_array((N, num_weight))
    Wbar = lil_array((N, num_weight))
    W[idx_r, np.arange(num_weight)] = 1
    Wbar[idx_c, np.arange(num_weight)] = 1
    W = W.tocsr()
    Wbar = Wbar.tocsr()
    weight_vec = val.T
    return W - Wbar, weight_vec

def mse_clusters(X: np.ndarray, y: np.ndarray, pred: np.ndarray, centers: np.ndarray) -> float:
    # N elements, d dimensions
    N, d = X.shape
    unique_preds = np.unique(pred)
    k = len(unique_preds)
    # make array for cluster centroids
    pred_centers = np.zeros((k, d))

    # for each predicted cluster
    for i, label in enumerate(unique_preds):
        # boolean mask that is True for rows where the predicted label matches the current label
        mask = pred == label
        # find the mean of points in the cluster, assign it to corresponding row in pred_centers
        pred_centers[i] = np.mean(X[mask], axis=0)

    # mapping of true cluster labels to their centroids
    true_centers = centers[y]
    # mapping of predicted cluster labels to their centroids
    pred_centers_mapped = pred_centers[pred]
    # find mse between the true and predicted cluster centroids
    # this is done by squaring the difference, summing over features, and then taking the mean over all data points
    mse = np.mean(np.sum((true_centers - pred_centers_mapped) ** 2, axis=1))

    return mse

def centroid_bound(X: npt.ArrayLike, gamma: float, lambd: float) -> float:
    N, d = X.shape
    if d == 1: d_prime = float('inf')
    if d == 2: d_prime = 4/3
    else: d_prime = d
    
    return gamma*N**(-1/max(d,2))*(np.log(N))**(1/d_prime)+(1+lambd)*gamma**(-1/3)

def trunc_bound()