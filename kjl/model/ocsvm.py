from sklearn.svm import OneClassSVM


class OCSVM(OneClassSVM):

    def __init__(self, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3, nu=0.5, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, random_state=100):
        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, nu=nu,
            shrinking=shrinking, cache_size=cache_size, verbose=verbose, max_iter=max_iter)

    # override decision_function. because test and grid_search will use decision_function first
    def decision_function(self, X):
        # it must be abnormal score because it will be used in grid search
        # we use y=1 as abnormal score, in grid search it will use y=1 as positive label,
        # so y_score also should be abnormal score
        # return -1 * self.score_samples(X)  # scores = sgn(model(x) - offset)
        return -1 * (self._decision_function(X).ravel() + self.offset_)

    def predict_proba(self, X):
        raise NotImplementedError
