import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'], dayfirst=True)
    indexNames = df[df['Temp'] < -20].index
    df.drop(indexNames, inplace=True)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df["Year"] = df["Year"].astype("category")
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('C:/Users/lbnow/Documents/huji/year3/IML/exes/ex2/city_temperature.csv')

    # Question 2 - Exploring data for specific country
    israel_df = df.loc[df['Country'] == 'Israel']
    fig = px.scatter(israel_df, x="DayOfYear", y="Temp", color="Year", title="The temperature in Israel according to months")
    fig.update_traces(marker=dict(size=3))
    fig.show()

    israel_months_std = (israel_df.groupby("Month").agg('std')).reset_index()
    (px.bar(israel_months_std, x="Month", y="Temp", title="The standard deviation of the daily temperatures for each month in Israel")).show()


    # Question 3 - Exploring differences between countries
    mu_std_counties = df.groupby(["Country", "Month"]).agg({"Temp": ['mean', 'std']})
    mu_std_counties.columns = [" ".join(x) for x in mu_std_counties.columns.ravel()]
    mu_std_counties = mu_std_counties.reset_index()
    mu_std_counties["Month"] = pd.to_numeric(mu_std_counties["Month"])
    mu_std_counties = mu_std_counties.sort_values("Month").reset_index(drop=True)
    (px.line(mu_std_counties, x="Month", y='Temp mean', color="Country", error_y="Temp std", title="The average and standard deviation of the temperature by months, divided to countries")).show()


    # Question 4 - Fitting model for different values of `k`
    X, y = israel_df['DayOfYear'], israel_df['Temp']
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    error_per_k = []
    for k in range(1, 11):
        poly_f = PolynomialFitting(k)
        poly_f._fit(train_X, train_y)
        k_loss = np.round(poly_f._loss(test_X, test_y), 2)
        print(k_loss)
        error_per_k.append([k, k_loss])
    error_per_k = pd.DataFrame(error_per_k, columns=['k value', 'MSE value'])
    (px.bar(error_per_k, x="k value", y="MSE value", title="MSE per k degree in poly-fit from day of year to temperature")).show()

    # Question 5 - Evaluating fitted model on different countries
    poly_f = PolynomialFitting(5)
    poly_f._fit(X, y)
    print()
    error_per_c = []
    for c in df.Country.unique():
        if c == 'Israel': continue
        country_df = df.loc[df['Country'] == c]
        X, y = country_df['DayOfYear'], country_df['Temp']
        c_loss = np.round(poly_f._loss(X, y), 2)
        error_per_c.append([c, c_loss])
    error_per_c = pd.DataFrame(error_per_c, columns=['Country', 'MSE value'])
    (px.bar(error_per_c, x="Country", y="MSE value",
            title="MSE per country in poly-fit from day of year to temperature")).show()