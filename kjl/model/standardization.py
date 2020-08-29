from sklearn.preprocessing import StandardScaler

# from examples.main_online_gmm import online_update_mean_variance
import numpy as np


class STD():

    def __init__(self):
        pass

    def fit(self, X_train):
        """

        Parameters
        ----------
        X_train

        Returns
        -------

        """
        self.n_samples, self.n_feats = X_train.shape

        self.scaler = StandardScaler()

        self.scaler.fit(X_train)

        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def update(self, x):
        # b) Update self.scaler
        self.scaler.mean_, self.scaler.scale_ = online_update_mean_variance(x, self.n_samples, self.scaler.mean_,
                                                                            self.scaler.scale_)
        self.scaler.var_ = np.square(self.scaler.scale_)
        # pass


def online_update_mean_variance(x, n, mu, sigma):
    """For standardization, we should online update mean and varicance (not covariance here)
    https://stackoverflow.com/questions/1346824/is-there-any-way-to-find-arithmetic-mean-better-than-sum-n

    Parameters
    ----------
    x : array-like, shape (1, n_feats)
    mu: array with shape (1, n_feats)
    sigma: array with shape (1, n_feats)

    Returns
    -------
        new_mu: array with shape (1, n_feats)
        new_sigma: array with shape (1, n_feats)
    """

    new_mu = mu + (x - mu) / (n + 1)
    new_sigma = sigma + (x - new_mu) * (x - mu)

    return new_mu, new_sigma
