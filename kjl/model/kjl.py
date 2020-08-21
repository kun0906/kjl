"""k-Jl methods
"""

# 1. standard libraries
import multiprocessing
import warnings

# 2. third-party packages
import numpy as np

# 3. your own package
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import pairwise_distances

# set some stuffs
from sklearn.utils import resample

warnings.simplefilter('always', FitFailedWarning)
multiprocessing.set_start_method('spawn', True)


# __all__= ['_grow_tree'] # allow private functions (start with _) can be imported by using "import *"

def getGaussianGram(Xrow, Xcol, sigma, goFast=1):
    """ get kernel (Gaussian) gram matrix
    The Gram matrix K is deﬁned as $K_ij = K(X_i , X_j) over a (sub) sample X = {X _i}, i=1,...,,n
    Parameters
    ----------
    Xrow
    Xcol
    sigma
    goFast

    Returns
    -------

    """
    if goFast == 1:
        A1 = np.expand_dims(np.power(np.linalg.norm(Xrow, axis=1), 2), axis=1)
        A2 = -2 * np.matmul(Xrow, np.transpose(Xcol))
        B = np.power(np.linalg.norm(Xcol, axis=1), 2)
        K = np.add(np.add(A1, A2), np.transpose(B))
        K = np.exp(-K * 1 / sigma ** 2)

    else:
        # Dist = np.linalg.norm(Xrow - Xcol)  # it's wrong!
        # Dist= cdist(Xrow, Xcol, metric='euclidean')
        Dist = pairwise_distances(Xrow, Y=Xcol, metric='euclidean')
        K = np.exp(-np.power(Dist, 2) * 1 / sigma ** 2)

    return K


def kernelJLInitialize(X, sigma, d, m, n, centering=0, independent_row_col=1, random_state=100):
    """Project data to d-dimension spaces

    Parameters
    ----------
    X
    sigma
    d
    m
    n
    centering
    independent_row_col

    Returns
    -------

    """
    print(f'random_state: {random_state}')
    np.random.seed(random_state)  # don't remove
    N, D = X.shape

    if independent_row_col:
        # # indRow and indCol are independent each other
        # indRow = np.random.randint(N, size=n)
        # indCol = np.random.randint(N, size=m)

        indRow = resample(range(N), n_samples=n, random_state=random_state,
                          replace=False)
        indCol = resample(range(N), n_samples=m, random_state=random_state,
                          replace=False)

    else:
        # Random select max(n,m) rows
        # indices = np.random.randint(N, size=max(n, m))
        indices = resample(range(N), n_samples=max(n, m), random_state=random_state,
                           replace=False)
        # In indRow and indCol, one includes another
        indRow = indices[0:n]
        indCol = indices[0:m]

    Xrow = X[indRow, :]  # nxD
    Xcol = X[indCol, :]  # mXD

    # compute Gaussian kernel gram matrix A (i.e., K generated from a subset of X)
    # print(Xrow, Xcol)
    A = getGaussianGram(Xrow, Xcol, sigma)  # nXm
    # print(A)
    if centering:
        # subtract the mean of col from each element in a col
        A = A - np.mean(A, axis=0)

    # Projection matrix: ZK (nXd = nXm * mXd) # matrix product : Gaussian sketching
    # U = np.dot(A, np.random.multivariate_normal([0] * d, np.diag([1] * d), m))
    random_matrix = np.random.multivariate_normal([0] * d, np.diag([1] * d), m)
    U = np.matmul(A, random_matrix)  # preferred for matrix multiplication
    print("Finished getting the projection matrix")

    # Obtain gram between full data and Xrow (Nxn)
    K = getGaussianGram(X, Xrow, sigma)

    # projected data (Nxd = NXn * nXd)
    KU = np.matmul(K, U)  # preferred for matrix multiplication
    print("Projected data")

    return KU, U, Xrow, random_matrix, A

# def merge_parameters(tuned_parameters):
#     if len(tuned_parameters.keys()) <=1:
#         return tuned_parameters
#     for i,
#
#     return
