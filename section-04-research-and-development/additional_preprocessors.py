import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin    

class TemporalVariableTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, variables, reference_variable):

        if not isinstance(variables, list):
            raise ValueError('Variables should be a list')

        self.variables = variables
        self.reference_variable = reference_variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()
        for var in self.variables:
            X[var] = X[self.reference_variable] - X[var]
        
        return X

class Mapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables, mappings):

        if not isinstance(variables, list):
            raise ValueError('Variables should be a list')

        if not isinstance(mappings, dict):
            raise ValueError('Mappings should be a dictionary')

        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()
        for var in self.variables:
            X[var] = X[var].map(self.mappings)

        return X

class ToObjectTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, variables):

        if not isinstance(variables, list):
            raise ValueError('Variables should be a list')

        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()
        for var in self.variables:
            X[var] = X[var].astype('O')

        return X