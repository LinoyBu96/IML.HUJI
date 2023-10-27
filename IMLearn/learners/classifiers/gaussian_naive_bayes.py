from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
        self.mu_ = np.empty((0, X.shape[
            1]))  # https://stackoverflow.com/questions/22392497/how-to-add-a-new-row-to-an-empty-numpy-array
        self.vars_ = np.empty((0, X.shape[1]))
        for k in range(K):
            k_samples = X[y == self.classes_[k],
                        :]  # https://stackoverflow.com/questions/16201536/select-rows-in-a-numpy-2d-array-with-a-boolean-vector
            k_mu = np.sum(k_samples, axis=0) / counts_per_class[k]
            k_mu = k_mu.reshape(1, k_mu.shape[0])
            self.mu_ = np.append(self.mu_, k_mu, axis=0)

            bar = k_samples - self.mu_[k].transpose()  # https://stackoverflow.com/questions/41991897/how-to-add-matrix-and-vector-column-wise
            k_var = (np.power(bar, 2)).sum(axis=0) / counts_per_class[k]
            k_var = k_var.reshape(1, k_var.shape[0])
            self.vars_ = np.append(self.vars_, k_var, axis=0)


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
        print(self.likelihood(X).shape)
        print(self.pi_.shape)
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

        for x in X:
            y_lh = []
            for y in self.classes_:
                dono = np.sqrt(2 * np.pi * self.vars_[y])
                m = np.exp(-0.5 * (np.power((x - self.mu_[y]), 2) / self.vars_[y])) / dono
                y_lh.append(np.prod(np.exp(-0.5 * (np.power((x - self.mu_[y]), 2) / self.vars_[y])) / dono))

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
