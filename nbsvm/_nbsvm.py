"""
sklearn interface to NBSVM classifier
"""

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import LinearSVC
from scipy.sparse.csr import csr_matrix
import scipy


class NBSVM(BaseEstimator, LinearClassifierMixin):
    """
    A NBSVM classifier following the sklearn API, as described in Section 2.3
    of Baselines and bigrams: simple, good sentiment and topic classification.
    https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf

    Parameters
    ----------
    alpha : float, default=1.
        Smoothing parameter for count vectors.
    beta : float, default=0.25
        Interpolation parameter between NB and SVM.
    C : float, default=1.
        Penalty parameter of the L2 error term for SVM.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    coef_ : array, shape = [1, n_features] if n_classes == 2
            else [n_classes, n_features]
        Weights assigned to the features, per sklearn.svm.LinearSVC.
    intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
        Constants in decision function, per sklearn.svm.LinearSVC.
    """

    def __init__(self, alpha=1.0, beta=0.25, C=1.0):
        self.alpha = alpha
        self.beta = beta
        self.C = C

    def fit(self, X, y):
        """
        Fit the NBSVM to a dataset.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray,
            shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate(X, y)

        self.classes_ = unique_labels(y)

        coefficients, intercepts = self._fit_one_model_per_class(X, y)

        self.coef_ = np.concatenate(coefficients)
        self.intercept_ = np.concatenate(intercepts)

        return self

    def _validate(self, X, y):
        """
        Validate that X and y are the correct shape, and that X contains no
        negative entries.
        """
        X, y = check_X_y(X, y, accept_sparse="csr")

        if scipy.sparse.issparse(X):
            self._validate_sparse(X)
        elif isinstance(X, np.ndarray):
            self._validate_dense(X)
        else:
            raise ValueError("""
                Not a scipy.sparse.csr.csr_matrix or numpy ndarray
            """)

        return X, y

    def _validate_sparse(self, X):
        if (X.data < 0.0).any():
            raise ValueError("All X entries should be non-negative")

    def _validate_dense(self, X):
        if (X < 0.0).any():
            raise ValueError("All X entries should be non-negative")

    def _fit_one_model_per_class(self, X, y):
        """
        Treat an n-class classification problem as n binary classification
        problems.
        """
        binary_models = [
            self._fit_binary_nbsvm(X, y == class_) for class_ in self.classes_
        ]
        coefficients, intercepts = zip(*binary_models)
        return coefficients, intercepts

    def _fit_binary_nbsvm(self, X, y):
        """
        Fit a NBSVM classifier to a binary classification problem.
        """
        r = self._log_count_ratio(X, y)

        X = X.multiply(r)

        svm = LinearSVC(C=self.C).fit(X, y)

        coef = self._interpolate(svm.coef_)
        coef *= r

        return coef, svm.intercept_

    def _log_count_ratio(self, X, y):
        """
        Log-count ratio computed from smoothed (by alpha) count vectors for
        each class. These are the coefficients in pure Multinomial Naive Bayes.
        """
        p = self.alpha + X[y == 1].sum(axis=0)
        q = self.alpha + X[y == 0].sum(axis=0)
        r = (self._log_normalize_count_vector(p) -
             self._log_normalize_count_vector(q))
        return r

    def _log_normalize_count_vector(self, arr):
        """
        Takes count vector and normalizes by L1 norm, then takes log.
        """
        return np.log(arr / np.linalg.norm(arr, 1))

    def _interpolate(self, coef):
        """
        Interpolate with parameter beta between Multinomial Naive Bayes
        (mean_weight) and SVM.
        """
        mean_weight = np.abs(coef).mean()
        return self.beta * coef + (1 - self.beta) * mean_weight
