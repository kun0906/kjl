import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import pairwise_distances

from kjl.model.kjl import getGaussianGram


class OCSVM(OneClassSVM):

    def __init__(self, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3, nu=0.5, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, random_state=100):
        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, nu=nu,
            shrinking=shrinking, cache_size=cache_size, verbose=verbose, max_iter=max_iter)

    # override decision_function. because test and grid_search will use decision_function first
    def decision_function_backup(self, X):
        # it must be abnormal score because it will be used in grid search
        # we use y=1 as abnormal score, in grid search it will use y=1 as positive label,
        # so y_score also should be abnormal score
        # return -1 * self.score_samples(X)  # scores = sgn(model(x) - offset)
        # return -1 * (self._decision_function(X).ravel() + self.offset_)
        return -1 * (self._decision_function(X).ravel())

    # override decision_function. try to use numpy.
    def decision_function(self, X):
        # pred_v =  coefficient * kernel(X, support_vectors) + intercept

        # self.dual_coef_: m x 1 (m is the number of support vectors)
        # self.support_vectors : m x D
        # kernel(X, support_vectors.T): nxm
        # X: n x D
        # self.intercept: 1x1
        # pred_v = kernel(X, support_vectors.T)  * self.dual_coef_  + self.intercept : nx1
        if self.kernel == 'rbf':
            # Dist = pairwise_distances(X, Y=self.support_vectors_, metric='euclidean')
            # # K = np.exp(-np.power(Dist, 2) * 1 / self.sigma ** 2)
            # K = np.exp(-self.gamma * np.power(Dist, 2))
            K = getGaussianGram(X, self.support_vectors_, 1/ np.sqrt(self.gamma))   # nxm
            pred_v = np.matmul(K, self.dual_coef_.transpose()).ravel() + self.intercept_
        elif self.kernel == 'linear':
            K = np.matmul(X, self.support_vectors_.transpose())
            pred_v = np.matmul(K, self.dual_coef_.transpose()).ravel() + self.intercept_
        else:
            raise NotImplementedError(self.kernel)

        # print(pred_v - (self._decision_function(X).ravel()))
        # return -1 * (pred_v + self.offset_)
        return -1 * pred_v

    def predict_proba(self, X):
        raise NotImplementedError
