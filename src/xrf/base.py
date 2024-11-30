"""ExPLAINABLE rANDOM fORESTS (xrf)

Classes implementing random forest classifiers and regressors with
example attribution, i.e., each prediction is associated with a weight
distribution over the training examples. The examples used in forming
a prediction can be limited by their number (k) or by their cumulative
weight (c).

Author: Henrik Boström (bostromh@kth.se)

Copyright 2024 Henrik Boström

License: BSD 3 clause
"""

__version__ = "0.1.1"

import numpy as np
from scipy.sparse import csr_array
from joblib import Parallel, delayed, cpu_count
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import normalize

class XRandomForestClassifier():
    """
    Explainable Random Forest Classifier.

    An explainable random forest classifier is generated in the same
    way as a standard random forest classifier, but provides example
    attributions, i.e., each prediction is associated with a weight
    distribution over the training examples, and allows for selecting
    a subset of the examples with the highest weight when forming
    predictions.
    
    The same set of parameters are available as for
    `sklearn.ensemble.RandomForestClassifier`
    """
    
    def __init__(self,**kwargs):
        kwargs.update({"oob_score": True})
        self.model = RandomForestClassifier()
        self.model.__dict__.update(kwargs)
        self.fitted = False
        self.time_fit = None
        self.classes_ = None
        self.y = None
        self.coverage = None
        
    def __repr__(self):
        return (f"XRandomForestClassifier(model={self.model}")

    def fit(self, X, y):
        """
        Fit explainable random forest classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            training objects
        y : array-like of shape (n_values,)
            training labels (numerical values)

        Returns
        -------
        self : object
            Fitted XRandomForestClassifier.
        """
        tic = time.time()
        self.model.fit(X, y)
        leaves = self.model.apply(X)
        coverage = []
        for i in range(len(self.model.estimators_)):
            bag = np.bincount(
                np.random.RandomState(self.model.estimators_[i].
                                      random_state).
                randint(0,len(X),len(X)),
                minlength=len(X))
            in_bag = bag > 0
            rows = leaves[:,i][in_bag]
            cols = np.arange(len(X))[in_bag]
            m = csr_array((bag[in_bag], (rows, cols)),
                          shape=(self.model.estimators_[i].
                                 tree_.node_count, len(X))).tolil()
            m = normalize(m, norm='l1')
            coverage.append(m)
        self.coverage = coverage
        class_indexes = np.array([np.argwhere(self.model.classes_ == y[i])[0][0] 
                                  for i in range(len(y))])
        self.y = np.zeros((len(y), len(self.model.classes_)))
        self.y[np.arange(len(y)), class_indexes] = 1
        self.fitted = True
        self.classes_ = self.model.classes_
        toc = time.time()
        self.time_fit = toc-tic
        return self
                
    def predict_proba(self, X, k=None, c=None, return_examples=False,
                      return_weights=False, normalize_weights=True):    
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            test objects
        k : no. of top-weighted training examples to use when forming
            predictions, default=None
        c : cumulative weight of top-weighted training examples to use when
            forming predictions, default=None
        return_examples : Boolean, default=False
            whether or not to output the indexes of training examples
            that are used when forming predictions
        return_weights : Boolean, default=False
            whether or not to output the weights of the training examples
            that are used when forming predictions (in decreasing order)
        normalize_weights : Boolean, default=True
            whether returned weights should be normalized or not

        Returns
        -------

        probabilities : ndarray of (n_samples,n_classes) with real values
            class probability distributions
        examples : ndarray of (n_samples, k) or (n_samples, ) of lists
            indexes of training examples used when forming predictions
            Only returned if return_examples == True.
        weights : ndarray of (n_samples, k) or (n_samples, ) of lists
            example weights used when forming predictions
            Only returned if return_weights == True.
        """
        leaves = self.model.apply(X)
        cdfs = np.array([self.coverage[i][leaves[:,i],:] for i in
                range(len(self.model.estimators_))])
        n_jobs = cpu_count()
        n_chunks, rem_chunks = divmod(len(self.model.estimators_), n_jobs)
        indexes = [(i*n_chunks+min(i, rem_chunks),
                    (i+1)*n_chunks+min(i+1, rem_chunks))
                   for i in range(n_jobs)]
        cdfs = Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(sum)(cdfs[ind[0]:ind[1]]) for ind in indexes)
        cdfs = sum(cdfs)
        cdfs = normalize(cdfs, norm='l1').toarray()
        weights = cdfs
        y_train = self.y
        if k is not None or c is not None:
            sorted_weights_indexes = np.flip(np.argsort(weights, axis=1), axis=1)
            if k is not None:
                top_indexes = sorted_weights_indexes[:, :k]
                weighted_predictions = normalize(
                    [np.dot(y_train[top_indexes[i]].T,
                            weights[i, top_indexes[i]])
                     for i in range(len(weights))], norm="l1")
            else:
                sorted_weights = np.array([weights[i, sorted_weights_indexes[i]]
                                           for i in range(len(weights))])
                cum_weights = np.cumsum(sorted_weights, axis=1)
                filtered_cum_weights = np.where(cum_weights<c, cum_weights, 
                                                np.inf)
                k = np.argmax(filtered_cum_weights, axis=1)+1
                top_indexes = np.array([sorted_weights_indexes[i, :k[i]] 
                                        for i in range(len(weights))],
                                       dtype="object")
                weighted_predictions = normalize(
                    [np.dot(y_train[top_indexes[i]].T, weights[i, top_indexes[i]])
                     for i in range(len(weights))], norm="l1")
        else: # k is None and c is None:
            weighted_predictions = np.array([np.dot(y_train.T, weights[i])
                                             for i in range(len(weights))])
        results = [weighted_predictions]
        if return_examples:
            if k is None and c is None:
                results.append(np.tile(np.arange(len(y_train)), (len(X),1)))
            else:
                results.append(top_indexes)
        if return_weights:
            if k is None and c is None:
                results.append(weights)
            else:
                weights = np.array([weights[i][top_indexes[i][weights[i][
                    top_indexes[i]] > 0]] for i in range(len(top_indexes))],
                                   dtype="object")
                if normalize_weights:
                    weights = np.array([w/np.sum(w) for w in weights],
                                       dtype="object")
                if len(weights.shape) > 1:
                    weights = weights.astype(float)
                results.append(weights)
        if len(results) == 1:
            return results[0]
        else:
            return results

    def predict(self, X, k=None, c=None, return_examples=False,
                return_weights=False, normalize_weights=True):    
        """
        Predict class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            test objects
        k : no. of top-weighted training examples to use when forming
            predictions, default=None
        c : cumulative weight of top-weighted training examples to use when
            forming predictions, default=None
        return_examples : Boolean, default=False
            whether or not to output the indexes of training examples
            that are used when forming predictions
        return_weights : Boolean, default=False
            whether or not to output the weights of the training examples
            that are used when forming predictions (in decreasing order)
        normalize_weights : Boolean, default=True
            whether returned weights should be normalized or not

        Returns
        -------

        labels : ndarray of (n_samples,) with class labels
            predicted classes
        examples : ndarray of (n_samples, k) or (n_samples, ) of lists
            indexes of training examples used when forming predictions
            Only returned if return_examples == True.
        weights : ndarray of (n_samples, k) or (n_samples, ) of lists
            example weights used when forming predictions
            Only returned if return_weights == True.
        """
        results = self.predict_proba(X, k, c, return_examples, return_weights,
                                     normalize_weights) 
        if isinstance(results, list):
            results[0] = np.array([self.model.classes_[np.argmax(results[0][i])]
                                   for i in range(len(results[0]))])
        else:
            results = np.array([self.model.classes_[np.argmax(results[i])]
                                   for i in range(len(results))])
        return results

class XRandomForestRegressor():
    """Explainable Random Forest Regressor.

    An explainable random forest regressor is generated in the same
    way as a standard random forest regressor, but provides example
    attributions, i.e., each prediction is associated with a weight
    distribution over the training examples, and allows for selecting
    a subset of the examples with the highest weight when forming
    predictions.
    
    The same set of parameters are available as for
    `sklearn.ensemble.RandomForestRegressor`
    """

    def __init__(self,**kwargs):
        kwargs.update({"oob_score": True})
        self.model = RandomForestRegressor()
        self.model.__dict__.update(kwargs)
        self.fitted = False
        self.time_fit = None
        self.y = None
        self.coverage = None
        
    def __repr__(self):
        return (f"XRandomForestRegressor(model={self.model}")

    def fit(self, X, y):
        """
        Fit explainable random forest regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            training objects
        y : array-like of shape (n_values,)
            training labels (numerical values)

        Returns
        -------
        self : object
            Fitted XRandomForestRegressor.
        """
        tic = time.time()
        self.model.fit(X, y)
        leaves = self.model.apply(X)
        coverage = []
        for i in range(len(self.model.estimators_)):
            bag = np.bincount(
                np.random.RandomState(self.model.estimators_[i].
                                      random_state).
                randint(0,len(X),len(X)),
                minlength=len(X))
            in_bag = bag > 0
            rows = leaves[:,i][in_bag]
            cols = np.arange(len(X))[in_bag]
            m = csr_array((bag[in_bag], (rows, cols)),
                          shape=(self.model.estimators_[i].
                                 tree_.node_count, len(X)))
            m = normalize(m, norm='l1')
            coverage.append(m)
        self.coverage = coverage
        self.y = y
        self.fitted = True
        toc = time.time()
        self.time_fit = toc-tic
        return self
                
    def predict(self, X, k=None, c=None, return_examples=False,
                return_weights=False, normalize_weights=True):    
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            test objects
        k : no. of top-weighted training examples to use when forming
            predictions, default=None
        c : cumulative weight of top-weighted training examples to use when
            forming predictions, default=None
        return_examples : Boolean, default=False
            whether or not to output the indexes of training examples
            that are used when forming predictions
        return_weights : Boolean, default=False
            whether or not to output the weights of the training examples
            that are used when forming predictions (in decreasing order)
        normalize_weights : Boolean, default=True
            whether returned weights should be normalized or not

        Returns
        -------

        predictions : ndarray of (n_samples,) with real values
            point predictions
        examples : ndarray of (n_samples, k) or (n_samples, ) of lists
            indexes of training examples used when forming predictions
            Only returned if return_examples == True.
        weights : ndarray of (n_samples, k) or (n_samples, ) of lists
            example weights used when forming predictions
            Only returned if return_weights == True.
        """
        leaves = self.model.apply(X)
        cdfs = np.array([self.coverage[i][leaves[:,i],:] for i in
                range(len(self.model.estimators_))])
        n_jobs = cpu_count()
        n_chunks, rem_chunks = divmod(len(self.model.estimators_), n_jobs)
        indexes = [(i*n_chunks+min(i, rem_chunks),
                    (i+1)*n_chunks+min(i+1, rem_chunks))
                   for i in range(n_jobs)]
        cdfs = Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(sum)(cdfs[ind[0]:ind[1]]) for ind in indexes)
        cdfs = sum(cdfs)
        cdfs = normalize(cdfs, norm='l1').toarray()
        weights = cdfs
        y_train = self.y                    
        if k is not None or c is not None:
            sorted_weights_indexes = np.flip(np.argsort(weights, axis=1),
                                             axis=1)
            if k is not None:
                top_indexes = sorted_weights_indexes[:, :k]
                weighted_predictions = np.sum([weights[i, top_indexes[i]] \
                                           * y_train[top_indexes[i]] \
                                           / np.sum(weights[i, top_indexes[i]])
                                           for i in range(len(top_indexes))], 
                                          axis=1)
            else:
                sorted_weights = np.array([weights[i, sorted_weights_indexes[i]]
                                           for i in range(len(weights))])
                cum_weights = np.cumsum(sorted_weights, axis=1)
                filtered_cum_weights = np.where(cum_weights<c, cum_weights, 
                                                np.inf)
                k = np.argmax(filtered_cum_weights, axis=1)+1
                top_indexes = [sorted_weights_indexes[i, :k[i]] 
                               for i in range(len(weights))]
                weighted_predictions = np.array([
                    np.sum(weights[i, top_indexes[i]] \
                           * y_train[top_indexes[i]] \
                           / np.sum(weights[i, top_indexes[i]]))
                    for i in range(len(top_indexes))])
        else:
            weighted_predictions = np.sum(weights*y_train, axis=1)
        results = [weighted_predictions]
        if return_examples:
            if k is None and c is None:
                results.append(np.tile(np.arange(len(y_train)), (len(X),1)))
            else:
                results.append(top_indexes)
        if return_weights:
            if k is None and c is None:
                results.append(weights)
            else:
                weights = np.array([weights[i][top_indexes[i][weights[i][
                    top_indexes[i]] > 0]] for i in range(len(top_indexes))],
                                   dtype="object")
                if normalize_weights:
                    weights = np.array([w/np.sum(w) for w in weights],
                                       dtype="object")
                if len(weights.shape) > 1:
                    weights = weights.astype(float)                
                results.append(weights)
        if len(results) == 1:
            return results[0]
        else:
            return results
