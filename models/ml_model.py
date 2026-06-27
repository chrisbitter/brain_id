import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from features import FeatureExtractor


def model_selection(model_name):
    """ Select the model based on the model name. """
    if model_name == 'svc':
        from sklearn.svm import SVC
        return SVC
    
    elif model_name == 'rfc':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier
    
    elif model_name == 'baseline':
        from models import BaselineClf
        return BaselineClf
    
    else:
        raise ValueError(f"Model {model_name} not found.")


class MLModel(BaseEstimator, ClassifierMixin):

    """ Class to combine the full model pipeline """

    def __init__(self, model_name='svc', feature_kwargs=None, model_kwargs=None):
        """ Initialize the model with the feature extractor and classifier. """
        self.feature_extractor = None
        self.model = None
        self.model_name = model_name
        self.model_cls = model_selection(model_name)
        self.feature_kwargs = feature_kwargs
        self.model_kwargs = model_kwargs
    
    def get_params(self, deep=True):
        return {
            # 'model_name': self.model_name,
            'feature_kwargs': self.feature_kwargs,
            'model_kwargs': self.model_kwargs
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        feature_kwargs = self.feature_kwargs or {}
        model_kwargs = self.model_kwargs or {}
        self.feature_extractor = FeatureExtractor(**feature_kwargs)
        self.model = self.model_cls(**model_kwargs)

        X = self.feature_extractor.fit_transform(X)
        self.model.fit(X, y)

    def predict(self, X):
        X = self.feature_extractor.transform(X)
        return self.model.predict(X)
    
