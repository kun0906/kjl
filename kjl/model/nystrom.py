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
from kjl.utils.data import data_info


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


class NYSTROM():
    def __init__(self, nystrom_params, debug=False):
        self.nystrom_params = nystrom_params
        self.debug = debug

    def fit(self, X_train):
        """Get Nystrom related data, such as

        Parameters
        ----------
        X_train

        Returns
        -------

        """
        if self.nystrom_params['nystrom']:
            d = self.nystrom_params['nystrom_d']
            n = self.nystrom_params['nystrom_n']
            q = self.nystrom_params['nystrom_q']

            start = datetime.now()
            n = n or max([200, int(np.floor(X_train.shape[0] / 100))])  # n_v: rows; m_v: cols. 200, 100?
            m = n
            if hasattr(self, 'sigma') and self.sigma:
                sigma = self.sigma
            else:
                # compute sigma
                dists = pairwise_distances(X_train)
                if self.debug:
                    # for debug
                    _qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                    _sigmas = np.quantile(dists, _qs)  # it will cost time
                    print(f'train set\' sigmas with qs: {list(zip(_sigmas, _qs))}')
                sigma = np.quantile(dists, q)
                if sigma == 0:
                    print(f'sigma:{sigma}, and use 1e-7 for the latter experiment.')
                    sigma = 1e-7
            self.sigma = sigma
            print("sigma: {}".format(sigma))

            # project train data
            self.Eigvec, self.Lambda, self.subX = nystromInitialize(X_train, self.sigma, n, d,
                                                                    random_state=self.random_state)
            # Phix = nystromFeatures(X_train, subX, sigma, Eigvec, Lambda)
            X_train = np.matmul(np.matmul(getGaussianGram(X_train, self.subX, self.sigma), self.Eigvec),
                                np.diag(1. / np.sqrt(np.diag(self.Lambda))))

            if self.debug: data_info(X_train, name='after nystrom, X_train')

            end = datetime.now()
            nystrom_train_time = (end - start).total_seconds()
            print("nystrom on train set took {} seconds".format(nystrom_train_time))


        else:
            nystrom_train_time = 0

        self.nystrom_train_time = nystrom_train_time

        # return X_train, subX, sigma, Eigvec, Lambda

        return self

    def transform(self, X):  # transform
        """Project X onto a lower space using Nystrom

        Parameters
        ----------
        X: array with shape (n_samples, n_feats)

        Returns
        -------
        X: array with shape (n_samples, d)
            "d" is the lower space dimension

        """
        if self.nystrom_params['nystrom']:
            # # for debug
            if self.debug:
                data_info(X, name='X_test_std')
                _qs = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
                _sigmas = np.quantile(pairwise_distances(X), _qs)
                print(f'test set\' sigmas with qs: {list(zip(_sigmas, _qs))}')

            start = datetime.now()
            print("Projecting test data")
            K = getGaussianGram(X, self.subX, self.sigma)  # get kernel matrix using rbf
            X = np.matmul(np.matmul(K, self.Eigvec), np.diag(1. / np.sqrt(np.diag(self.Lambda))))
            if self.debug: data_info(X, name='after nystrom, X_test')
            end = datetime.now()
            nystrom_test_time = (end - start).total_seconds()
            print("nystrom on test set took {} seconds".format(nystrom_test_time))
        else:
            nystrom_test_time = 0

        self.nystrom_test_time = nystrom_test_time

        return X

    def update(self):

        raise NotImplementedError('error')


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
