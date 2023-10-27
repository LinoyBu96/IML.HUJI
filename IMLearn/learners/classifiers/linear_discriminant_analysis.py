from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m = y.shape[0]  # number of samples
        self.classes_, counts_per_class = (np.unique(y, return_counts=True))  # returns as ascending order of classes
        K = self.classes_.shape[0]  # number of classes
        self.pi_ = counts_per_class / m
        self.mu_ = np.empty((0, X.shape[1]))  # https://stackoverflow.com/questions/22392497/how-to-add-a-new-row-to-an-empty-numpy-array
        for k in range(K):
            k_samples = X[y == self.classes_[k], :]  # https://stackoverflow.com/questions/16201536/select-rows-in-a-numpy-2d-array-with-a-boolean-vector
            k_mu = np.sum(k_samples, axis=0) / counts_per_class[k]
            k_mu = k_mu.reshape(1, k_mu.shape[0])
            self.mu_ = np.append(self.mu_, k_mu, axis=0)
        bar = X - self.mu_[y, :]  # https://stackoverflow.com/questions/32191029/getting-the-indices-of-several-elements-in-a-numpy-array-at-once
        self.cov_ = np.dot(np.transpose(bar), bar) / (m - K)
        self._cov_inv = np.linalg.inv(self.cov_)


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
        return np.argmax(self.likelihood(X) * self.pi_.reshape(1, self.pi_.shape[0]), axis=1)


    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        d = X.shape[1]
        multi_gaus = []
        dono = np.sqrt(np.power(2 * np.pi, d) * np.linalg.det(self.cov_))

        for x in X:
            y_lh = []
            for y in self.classes_:
                y_lh.append(
                    np.exp(-0.5 * ((x - self.mu_[y]) @ np.linalg.inv(self.cov_) @ np.transpose(x - self.mu_[y]))) / dono)
            multi_gaus.append(y_lh)

        return np.array(multi_gaus)


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from loss_functions import misclassification_error
        return misclassification_error(y, self._predict(X), False)
