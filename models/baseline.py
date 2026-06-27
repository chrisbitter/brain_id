import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class BaselineClf(BaseEstimator, ClassifierMixin):

    """
    Baseline classifier that predicts the most frequent class in the training set.
    This is a simple classifier that can be used as a baseline to compare with other classifiers.
    """

    def __init__(self):
        self.best_y = 0
    
    def fit(self, X, y):

        # get the most frequent class
        self.best_y = np.bincount(y).argmax()

    def predict(self, X):
        return np.full(X.shape[0], self.best_y)