from datetime import datetime

from sklearn import metrics
from sklearn.metrics import average_precision_score, roc_curve

from kjl.utils.data import dump_data


class BASE_MODEL:

    def __init__(self):
        pass

    def _train(self, model, X_train, y_train=None):
        """Train model on the (X_train, y_train)

        Parameters
        ----------
        model
        X_train
        y_train

        Returns
        -------

        """
        start = datetime.now()
        try:
            model.fit(X_train)
        except Exception as e:
            msg = f'fit error: {e}'
            print(msg)
            raise ValueError(f'{msg}: {model.get_params()}')
        end = datetime.now()
        train_time = (end - start).total_seconds()
        print("Fitting model takes {} seconds".format(train_time))

        return model, train_time

    def _test(self, model, X_test, y_test):
        """Evaulate the model on the X_test, y_test

        Parameters
        ----------
        model
        X_test
        y_test

        Returns
        -------
           y_score: abnormal score
           testing_time, auc, apc
        """
        start = datetime.now()
        # For inlier, a small value is used; a larger value is for outlier (positive)
        # it must be abnormal score because we use y=1 as abnormal and roc_acu(pos_label=1)
        y_score = model.decision_function(X_test)

        """
        if model_name == "Gaussian" and n_components != 1:
            preds = model.predict_proba(X_test)
            pred = 1 - np.prod(1-preds, axis=1)
        else:
            pred = model.score_samples(X_test)
        """
        end = datetime.now()
        testing_time = (end - start).total_seconds()
        print("Test model takes {} seconds".format(testing_time))

        apc = average_precision_score(y_test, y_score, pos_label=1)
        # For binary  y_true, y_score is supposed to be the score of the class with greater label.
        # auc = roc_auc_score(y_test, y_score)  # NORMAL(inliers): 0, ABNORMAL(outliers: positive): 1
        # pos_label = 1, so y_score should be the corresponding score (i.e., abnormal score)
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # auc1 = roc_auc_score(y_test, y_score)
        # print(model.get_params())
        # assert auc==auc1

        # f1, bestEp = selectThreshHold(test_y_i, pred)

        # if auc > max_auc:
        #     max_auc = auc
        #     best_pred = y_score

        print("APC: {}".format(apc))
        print("AUC: {}".format(auc))
        # print("F1: {}".format(f1))

        return y_score, testing_time, auc

    def test(self, X_test, y_test):

        start = datetime.now()
        # For inlier, a small value is used; a larger value is for outlier (positive)
        # it must be abnormal score because we use y=1 as abnormal and roc_acu(pos_label=1)
        y_score = self.model.decision_function(X_test)

        """
        if model_name == "Gaussian" and n_components != 1:
            preds = model.predict_proba(X_test)
            pred = 1 - np.prod(1-preds, axis=1)
        else:
            pred = model.score_samples(X_test)
        """
        end = datetime.now()
        testing_time = (end - start).total_seconds()
        print("Test model takes {} seconds".format(testing_time))

        apc = average_precision_score(y_test, y_score, pos_label=1)
        # For binary  y_true, y_score is supposed to be the score of the class with greater label.
        # auc = roc_auc_score(y_test, y_score)  # NORMAL(inliers): 0, ABNORMAL(outliers: positive): 1
        # pos_label = 1, so y_score should be the corresponding score (i.e., abnormal score)
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # auc1 = roc_auc_score(y_test, y_score)
        # print(model.get_params())
        # assert auc==auc1

        # f1, bestEp = selectThreshHold(test_y_i, pred)

        # if auc > max_auc:
        #     max_auc = auc
        #     best_pred = y_score

        print("APC: {}".format(apc))
        print("AUC: {}".format(auc))
        # print("F1: {}".format(f1))

    def save(self, data, out_file='.dat'):
        dump_data(data, name=out_file)
