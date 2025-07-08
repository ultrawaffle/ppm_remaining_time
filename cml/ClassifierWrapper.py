import numpy as np
from numpy import array
import xgboost as xgb
#import cupy


class ClassifierWrapper(object):

    def __init__(self, cls, min_cases_for_training=30, eval_set=None):
        """
        Constructor for the ClassifierWrapper class.
        Parameters
        ----------
        cls: RegressorMixin
            The classifier to wrap.
        min_cases_for_training: int
            The minimum number of cases required for training. Standard is 30.
        """
        self.cls = cls
        self.min_cases_for_training = min_cases_for_training
        self.hardcoded_prediction = None
        self.eval_set = eval_set

    def fit(self, X, y):
        """
        Fit the classifier.
        Parameters
        ----------
        X: pd.DataFrame
        y: pd.Series

        Returns
        -------
        object
            The fitted ClassifierWrapper.
        """
        #print(X)
        # if there are too few training instances, use the mean
        if X.shape[0] < self.min_cases_for_training:
            self.hardcoded_prediction = np.median(y) # predict median since MAE optimizes for median

        # if all the training instances are of the same class, use this class as prediction
        elif len(set(y)) < 2:
            self.hardcoded_prediction = int(y.values[0])

        else:
            # y = cupy.asarray(y.values.astype(np.float32))
            # X = cupy.asarray(X.astype(np.float32))
            #print(X)
            y = np.asarray(y.values.astype(np.float32))
            X = np.asarray(X.astype(np.float32))

            #evals = [(xgb.DMatrix(eval_set[0], label=eval_set[1]), 'validation')]

            self.cls.fit(np.asarray(X.astype(np.float32)), 
                         np.asarray(y.astype(np.float32)),
                         #eval_set=self.eval_set, 
                         #verbose=False 
                         )
            # self.cls.fit(cupy.asarray(X.astype(np.float32)), cupy.asarray(y.astype(np.float32)))
            return self

    def predict_proba(self, X, y=None):
        """
        Predict prediction for each sample.
        Parameters
        ----------
        X: pd.DataFrame
        y

        Returns
        -------
        np.array
            The prediction for each sample.

        """
        if self.hardcoded_prediction is not None:
            return array([self.hardcoded_prediction] * X.shape[0])

        else:
            # preds = self.cls.predict(cupy.asarray(X.astype(np.float32)))
            preds = self.cls.predict(np.asarray(X.astype(np.float32)))
            return preds

    def fit_predict(self, X, y):
        """
        Fit and predict the prediction for each sample.
        Parameters
        ----------
        X: pd.DataFrame
        y

        Returns
        -------
        np.array
            The prediction for each sample.
        """

        self.fit(X, y)
        return self.predict_proba(X)
