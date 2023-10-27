import math
from typing import Tuple
import numpy as np
import pandas as pd




class BaseEstimator(ABC):
    """
    Base class of supervised estimators (classifiers and regressors)
    """

    def __init__(self) -> BaseEstimator:
        """
        Initialize a supervised estimator instance

        Attributes
        ----------
        fitted_ : bool
            Indicates if estimator has been fitted. Set by ``self.fit`` function
        """
        self.fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """
        Fit estimator for given input samples and responses

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        After fitting sets ``self.fitted_`` attribute to `True`
        """
        self._fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Raises
        ------
        ValueError is raised if ``self.predict`` was called before calling ``self.fit``
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling ``predict``")
        return self._predict(X)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function specified for estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function specified for estimator

        Raises
        ------
        ValueError is raised if ``self.loss`` was called before calling ``self.fit``
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling ``loss``")
        return self._loss(X, y)

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit estimator for given input samples and responses

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        raise NotImplementedError()

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        raise NotImplementedError()

    @abstractmethod
    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function specified for estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function specified for estimator
        """
        raise NotImplementedError()

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit an estimator over given input data and predict responses for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        self.fit(X, y)
        return self.predict(X)




def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.
    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.
    train_proportion: Fraction of samples to be split as training set
    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set
    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples
    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set
    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    inds = np.random.choice(y.shape[0], size=math.ceil(X.shape[0] * train_proportion), replace=False)
    labels_to_remove = (y.keys())[inds]
    #print("ok", labels_to_remove)
    #X.iloc[inds]
    #y.iloc[inds]

    return X.iloc[inds], y.iloc[inds], X.drop(labels_to_remove, axis=0), y.drop(labels_to_remove)


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors
    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers
    b: ndarray of shape (n_samples,)
        Second vector of integers
    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()

