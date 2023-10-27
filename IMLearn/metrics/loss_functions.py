import numpy as np

def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    return np.sum((y_true - y_pred) ** 2) / y_true.shape[0]  # checked and probably correct


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    #misclassification = np.sum(np.abs(y_pred - y_true)) / 2  # if y_p == y_t ==> 0, else ==> 2/-2
    misclassification = (y_true != y_pred).sum()
    return misclassification / y_true.shape[0] if normalize else misclassification


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    FP_TN = misclassification_error(y_true, y_pred, False)
    TP_TN = y_true.shape - FP_TN
    return (TP_TN/y_true.shape)[0]

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    return - np.sum(y_true * np.log(y_pred))

def softmax(X: np.ndarray) -> np.ndarray:
    """
    Compute the Softmax function for each sample in given data
    Parameters:
    -----------
    X: ndarray of shape (n_samples, n_features)
    Returns:
    --------
    output: ndarray of shape (n_samples, n_features)
        Softmax(x) for every sample x in given data X
    """
    es = np.exp(X)
    sums = np.sum(es, axis=1)
    return es / sums[:, None]