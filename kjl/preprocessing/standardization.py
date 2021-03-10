# from examples.main_online_gmm import online_update_mean_variance
import numpy as np
from sklearn.preprocessing import StandardScaler


class STD():

    def __init__(self, with_means=True):
        self.with_means = with_means

    def fit(self, X_train):
        """

        Parameters
        ----------
        X_train

        Returns
        -------

        """
        self.n_samples, self.n_feats = X_train.shape

        self.scaler = StandardScaler(with_mean=self.with_means)

        self.scaler.fit(X_train)

        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def update(self, x):
        # Update self.scaler, in which mean_ is mean, scale_ is np.sqrt(variance/n-1)
        self.scaler.mean_, self.scaler.scale_ = online_update_mean_variance(x, self.n_samples, self.scaler.mean_,
                                                                            self.scaler.scale_)
        self.scaler.var_ = np.square(self.scaler.scale_)
        self.n_samples += x.shape[0]


def online_update_mean_variance(x, n, mu, sigma):
    """For standardization, we should online update mean and varicance (not covariance here)
    https://stackoverflow.com/questions/1346824/is-there-any-way-to-find-arithmetic-mean-better-than-sum-n
    https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Parameters
    ----------
    x : array-like, shape (n_samples, n_feats)
    mu: array with shape (1, n_feats)
    sigma: array with shape (1, n_feats)
        sigma = np.sqrt(variance/n-1)

    Returns
    -------
        new_mu: array with shape (1, n_feats)
        new_sigma: array with shape (1, n_feats)
    """
    # sigma_sq = sigma ** 2 * (n-1)
    # for _x in x:
    #     mu_prev = mu
    #     mu += (_x - mu) / (n + 1)
    #     sigma_sq += (_x - mu) * (_x - mu_prev)   # C = sigma_sq*(n-1)
    #     n += 1
    #
    # new_mu = mu
    # new_sigma = np.sqrt(sigma_sq / (n - 1))
    # print(new_mu, new_sigma)

    m = x.shape[0]
    new_mu = mu + np.sum(x - mu[np.newaxis, :], axis=0) / (n + m)

    # *: element product
    C = np.sum((x - new_mu[np.newaxis, :]) * (x - mu[np.newaxis, :]), axis=0)
    new_sigma = np.sqrt((sigma ** 2 * (n - 1) + C) / (n + m - 1))
    # print(f'mu: {mu}, sigma: {sigma}')
    # print(f'new_mu: {new_mu}, new_sigma: {new_sigma}')

    return new_mu, new_sigma
