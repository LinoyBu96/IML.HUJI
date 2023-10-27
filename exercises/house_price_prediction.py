from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import os 
pio.templates.default = "simple_white"



def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df[df.bedrooms != 0]

    indexNames = df[df['price'] <= 0].index
    df.drop(indexNames, inplace=True)
    df = df.dropna()

    df["yr_renovated"] = df.apply(lambda x: x["yr_renovated"] if x["yr_renovated"] !=0 else x["yr_built"], axis=1)

    df = pd.get_dummies(df, columns=['zipcode'])

    df = df.drop(['id', 'date'], axis=1)
    return df




def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for f in X.columns:
        pearsonr = np.cov(X[f], y)[0][1] / (np.std(X[f]) * np.std(y))
        fig = go.Figure(go.Scatter(x=X[f], y=y, mode='markers', name=r'$\widehat\mu$'),
                  layout=go.Layout(
                      title=r"$\text{The correlation between '} \text{" + f + "} \\text{' feature to the price Pearson Correlation = } \\text{"+ str(round(pearsonr, 3)) +"}$",
                      xaxis_title="$\\text{'" + f + "'} \\text{ feature value}$",
                      yaxis_title="r$\\text{the house price}$",
                      height=350, width=750))
        fig.write_image(output_path + "/" + f + ".png")



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df = load_data('C:/Users/lbnow/Documents/huji/year3/IML/exes/ex2/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    X, y = df.loc[:, df.columns != 'price'], df['price']
    feature_evaluation(X, y, 'figs')

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)



    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    lg = LinearRegression(True)
    train_data = train_X
    train_data['price'] = train_y

    mean_loss = []
    std_loss = []
    for p in range(10, 101):
        p_loss = []
        for i in range(10):
            sample = train_data.sample(frac=p/100, replace=False)
            sample_X, sample_y = sample.loc[:, sample.columns != 'price'], sample['price']
            lg._fit(sample_X, sample_y)
            p_loss.append(lg._loss(test_X, test_y))
        mean_loss.append(np.mean(p_loss))
        std_loss.append(np.std(p_loss))


    fig = go.Figure(data=((go.Scatter(x=np.arange(10, 101), y=mean_loss, mode='markers+lines', name="Mean prediction")),
                     (go.Scatter(x=np.arange(10, 101), y=(np.array(mean_loss) - 2 * np.array(std_loss)), fill=None, mode="lines", line=dict(color="lightgrey"),
                                showlegend=False)),
                     (go.Scatter(x=np.arange(10, 101), y=(np.array(mean_loss) + 2 * np.array(std_loss)), fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                                showlegend=False))),
              layout=go.Layout(
                  title='The average loss as function of training size with error ribbon of size',
                  xaxis_title="Percentage of training data size",
                  yaxis_title="Mean value",
                  height=500))
    fig.show()
