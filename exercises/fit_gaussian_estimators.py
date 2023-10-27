from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    m = 1000
    mu = 10
    sigma = 1
    samples = np.random.normal(mu, sigma, m)
    ug = UnivariateGaussian()
    ug.fit(samples)  # fits estimated mu and sigma^2 (var)
    print("(" + str(ug.mu_) + ", " + str(ug.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, m, int(m / 10)).astype(int)  # vector of [10, 20, 30, ..., 1000]
    estimated_dis = []
    for m in ms:
        X = np.random.normal(mu, sigma, size=m)
        estimated_dis.append(abs(mu - np.mean(X)))  # absolute distance between estimated mu to real mu

    go.Figure(go.Scatter(x=ms, y=estimated_dis, mode='markers+lines', name=r'$\widehat\mu$'),
              layout=go.Layout(
                  title=r"$\text{Absolute Distance Between The Estimated And True Value Of The Expectation As Function Of Number Of Samples}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="r$|\hat\mu - \mu|$",
                  height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    samplesPDF = ug.pdf(samples)
    go.Figure(go.Scatter(x=samples, y=samplesPDF, mode='markers', name=r'$\widehat\mu$'),
              layout=go.Layout(
                  title=r"$\text{The empirical PDF function under the fitted model of } N(\mu, {\sigma^2})$",
                  xaxis_title="$\\text{sample values}$",
                  yaxis_title="$\\text{PDF values}$",
                  height=300, width=700)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = [0, 0, 4, 0]
    cov = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    mSamples = np.random.multivariate_normal(mean, cov, 1000)
    mg = MultivariateGaussian()
    mg.fit(mSamples)
    print(mg.mu_)
    print(mg.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    values = []
    for c in f3:
        values_vec = []
        for a in f1:
            values_vec.append(mg.log_likelihood([a, 0, c, 0], cov, mSamples))
        values.append(values_vec)

    go.Figure(go.Heatmap(x=f1, y=f3, z=values, colorbar=dict(title="logL")),
              layout=go.Layout(title=r"$\text{Log-Likelihood Heatmap for mean = [f1, 0, f3, 0]}$",
                               xaxis_title="$\\text{f1 values}$",
                               yaxis_title="$\\text{f3 values}$", height=600, width=600)).show()


    # Question 6 - Maximum likelihood
    ind = np.unravel_index(np.argmax(values, axis=None), np.shape(values))
    print("maximum log-likelihood values: f1 = " + str(round(f1[ind[1]], 4)) + ", f3 = " + str(round(f3[ind[0]], 4)))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
