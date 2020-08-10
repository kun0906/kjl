import pickle
import random
from collections import Counter

import scipy
import scipy.linalg as la
import numpy as np
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
from datetime import datetime

from sklearn.metrics import pairwise_distances
from sklearn.utils import resample, shuffle

from kjl.model.kjl import getGaussianGram


def nystromInitialize(Xtrain, sigma, n, d, random_state=42):
    """
function [Eigvec,Lambda,subX]= nystromInitialize(X,sigma,n, d)
%   X = N by D data matrix (training data)
%   sigma = Gaussian kernel variance
%   n less than N (n is the subsample size)
%   d less than N (d is the number of random features)
%   NOTE: default value for d is n

%Returns the following which are useful to compute the feature map
% subX = n by D subsampled matrix
% Eigvec = top d eigen vector matrix associated with the Gram matrix of subX (n by d matrix)
% Lambda = (d by d) top d diagonal eigen value matrix associated with the Gram matrix of subX (n by n matrix)

if nargin < 4
    d = n;
end

%rng(1); % for reproducibility, reset random number generator;

[N,D] = size(X);

%uniformly subsample n points from training data
temp=randperm(N);
subX=X(temp(1:n),:);

%Compute the kernel matrix for the subsampled data
Ksub=getGaussianGram(subX,subX,sigma);

%eigdecomposition of the kernel matrix of subsampled data
[Eigvec,Lambda] = eigs(Ksub, d);

end

    Parameters
    ----------
    Xtrain
    n
    d
    q

    Returns
    -------

    """
    rows, cols = Xtrain.shape

    # %uniformly subsample n points from training data
    # temp = np.random.permutation(rows)
    # subX = Xtrain[temp[1:n], :]
    temp = np.random.RandomState(seed=random_state).permutation(rows)
    subX = Xtrain[temp[0:n], :]
    # subX = shuffle(Xtrain, n_samples=n, random_state=random_state)  # use resample under the hood

    # %Compute the kernel matrix for the subsampled data
    Ksub = getGaussianGram(subX, subX, sigma)

    # %eigdecomposition of the kernel matrix of subsampled data
    # Eigvec, Lambda = eigs(Ksub, d)
    # use v0 = np.ones(Ksub.shape[0]) to fix the result
    Lambda, Eigvec = scipy.sparse.linalg.eigs(Ksub, k=d, which='LM', v0=np.ones(Ksub.shape[0]))
    Lambda = np.real(np.diag(Lambda))  # np.diag(Lambda) to make it has the same format with matlab output
    Eigvec = np.real(Eigvec)

    return Eigvec, Lambda, subX


def nystromFeatures(Xtrain, subX, sigma, Eigvec, Lambda):
    """
function [Phix]= nystromFeatures(X,subX,sigma,Eigvec,Lambda)

% Return feature map and projected data
%   X = N by D data matrix
%  subX = n by D data matrix (subsampled from X)
%   sigma = Gaussian kernel variance
%   d less than N
% Eigvec = eigen vector matrix associated with the Gram matrix of subX (n by d matrix)
% Lambda = diagonal eigen value matrix associated with the Gram matrix of subX (d by d matrix)
%  Phix is the feature map $Nxd$

%the feature map Phi(X)=Lambda^{-1/2}*Eigvec'*(K(X,X_i))^T_i for i=1..N

Phix=getGaussianGram(X,subX,sigma)*Eigvec*diag(1./sqrt(diag(Lambda)));

end


    Parameters
    ----------
    Xtrain
    subX
    sigma
    Eigvec
    Lambda

    Returns
    -------

    """

    # %the feature map Phi(X)=Lambda^{-1/2}*Eigvec'*(K(X,X_i))^T_i for i=1..N
    # Phix = getGaussianGram(Xtrain, subX, sigma)*Eigvec * np.diag(1. / np.sqrt(np.diag(Lambda)))
    Phix = np.matmul(np.matmul(getGaussianGram(Xtrain, subX, sigma), Eigvec), np.diag(1. / np.sqrt(np.diag(Lambda))))

    return Phix


# def nystrom_proj(Xtrain, n, d, q, random_state=42):
#     ## Get Nystrom features
#     start = datetime.now()
#     dists = pairwise_distances(Xtrain)
#     sigma = np.quantile(dists, q=q)
#     Eigvec, Lambda, subX = nystromInitialize(Xtrain, sigma, n, d, random_state=random_state)
#     PhiX = nystromFeatures(Xtrain, subX, sigma, Eigvec, Lambda)
#     end = datetime.now()
#     nystrom_training_time = (end - start).total_seconds()
#     print("nystrom on test set took {} seconds".format(nystrom_training_time))
#
#     return PhiX


def nystrom_cluster(Xtrain, k, sigma, n, d, random_state=42):
    """
function [idx PhiX feattime] = NystromCluster(Xtrain, k, sigma, n, d)

%% Return k clusters by kernel JL, n subsamples, feature dimension d
% Xtrain: N data points in D dimensions
% idx: cluster assignments

[N, D] = size(Xtrain);

idx = [];

defaultQuant = .25; % to be used to choose sigma

switch nargin
    case 3
        n = floor(N/10);
        d = k;
    case 4
        d = k;
end


%% Get Nystrom features
tic;
[Eigvec,Lambda,subX]= nystromInitialize(Xtrain,sigma,n, d);
[PhiX]= nystromFeatures(Xtrain,subX,sigma,Eigvec,Lambda);
feattime = toc;

%size(PhiX)

%% Cluster

idx = kmeans(PhiX, k);

    Returns
    -------
        Return k clusters by kernel JL, n subsamples, feature dimension d
        Xtrain: N data points in D dimensions
        idx: cluster assignments
    """

    rows, cols = Xtrain.shape
    if not n:
        n = np.floor(rows / 10)

    d = k

    # defaultQuant = .25; % to be used to choose sigma

    ## Get Nystrom features
    start = datetime.now()
    Eigvec, Lambda, subX = nystromInitialize(Xtrain, sigma, n, d)
    PhiX = nystromFeatures(Xtrain, subX, sigma, Eigvec, Lambda)
    end = datetime.now()
    nystrom_training_time = (end - start).total_seconds()
    print("nystrom on test set took {} seconds".format(nystrom_training_time))

    print(f'PhiX: {PhiX.shape} {PhiX}')
    # % % Cluster
    km = KMeans(n_clusters=k, random_state=random_state)
    idx = km.fit_predict(PhiX)
    print(f'idx: {idx}')
    return idx


def load_data(in_file):
    with open(in_file, 'rb') as f:
        data = pickle.load(f)

    return data


def main():
    normal_file = 'data/data_kjl/DS10_UNB_IDS/DS11-srcIP_192.168.10.5/iat_size/header:False/normal.dat'

    normal_data = load_data(normal_file)
    X_train = normal_data[:5000, :]
    dists = pairwise_distances(X_train)
    sigma = np.quantile(dists, q=0.25)
    n = 100
    d = 10
    k = d

    idx = nystrom_cluster(X_train, k, sigma, n, d)
    print(Counter(idx))


if __name__ == '__main__':
    main()
