import Perceptron
from linear_discriminant_analysis import LDA
from gaussian_naive_bayes import GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import numpy as np
from sklearn.linear_model import Perceptron as pp #to del
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class
    Parameters
    ----------
    filename: str
        Path to .npy data file
    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used
    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class
    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets
    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset('C:/Users/lbnow/Downloads/important/IML/IML.HUJI/datasets/' + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        def loss_count_callback(fit: Perceptron, x: np.ndarray, y_i: int):
            losses.append(fit._loss(X, y))

        a = Perceptron(True, 1000, loss_count_callback).fit(X, y)

        # Plot figure of loss as function of fitting iteration'
        go.Figure(go.Scatter(x=np.arange(len(losses)), y=losses, name=r'$\widehat\mu$'),
                  layout=go.Layout(
                      title=rf"$\textbf{{{n} Data: Misclassification Loss As A Function Of Nu. of Iteration of Perceptron Classifier}}$",
                      xaxis_title=rf"$\textbf{{Nu. of Iteration}}$",
                      yaxis_title=rf"$\textbf{{Misclassification Loss}}$", height=350, width=1000)).show()

        """los_pp = []
        from loss_functions import misclassification_error
        for i in range(1, 1001):
            b = pp(fit_intercept=True, max_iter=i)
            b.fit(X, y)
            g = misclassification_error(y, b.predict(X), False)
            print(g)
            los_pp.append(g)

        go.Figure(go.Scatter(x=np.arange(len(los_pp)), y=los_pp, name=r'$\widehat\mu$')).show()"""





def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def get_ellipse2(mu: np.ndarray, vars: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(vars)[::-1])
    theta = atan2(l1 - vars[0, 0], vars[0, 1]) if vars[0, 1] != 0 else (np.pi / 2 if vars[0, 0] < vars[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:

        # Load dataset
        X, y = load_dataset('C:/Users/lbnow/Downloads/important/IML/IML.HUJI/datasets/' + f)

        # Fit models and predict over training set
        models = [LDA(), GaussianNaiveBayes()]
        model_names = ["LDA", "Gaussian Naive Bayes"]
        models[0].fit(X, y)
        models[1].fit(X, y)
        from IMLearn.metrics import accuracy
        accuracies = [accuracy(y, models[0].predict(X)), accuracy(y, models[1].predict(X))]

        fig = make_subplots(rows=1, cols=2, subplot_titles=[rf"$\textbf{{{m} Estimator, Accuracy = {p}}}$" for m, p in zip(model_names, accuracies)],
                            horizontal_spacing=0.05, vertical_spacing=.03)

        for i, m in enumerate(models):
            #lda = LinearDiscriminantAnalysis(store_covariance=True)

            # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
            # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
            # Create subplots

            """plot_step = 0.02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            x_ = np.arange(x_min, x_max, plot_step)
            y_ = np.arange(y_min, y_max, plot_step)
            xx, yy = np.meshgrid(x_, y_)
    
            Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
    
            #cmap = matplotlib_to_plotly(plt.cm.Paired, 5)
            cs = go.Contour(x=x_, y=y_, z=Z, showscale=False)"""

            # Add traces for data-points setting symbols and colors
            symbols = np.array(["circle", "square", "star"])
            fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=m.predict(X), symbol=symbols[y],
                                                   line=dict(color="black", width=1)))],
                           rows=(0) + 1, cols=(i)+1)



            # Add `X` dots specifying fitted Gaussians' means
            fig.add_trace(go.Scatter(x=m.mu_[:, 0], y=m.mu_[:, 1], mode="markers", showlegend=False,
                                     marker=dict(size=10, symbol="x", color="black")))

            # Add ellipses depicting the covariances of the fitted Gaussians
            if i == 0:
                for mu in m.mu_:
                    fig.add_trace(get_ellipse(mu, m.cov_))
            """else:
                for mu in m.mu_:
                    fig.add_trace(get_ellipse2(mu, m.vars_))"""
        fig.update_layout(title=rf"$\textbf{{Predictions of {f} Dataset}}$",
                          margin=dict(t=100))
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()