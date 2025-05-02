import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Load the dataset

data = pd.read_csv('mnist.csv')
data.head()

# Prepare the data
arrayList = data.to_numpy()
len(arrayList)
X_train = arrayList[0:41, 0]
Y_train = arrayList[0:41, 1]
x_test = arrayList[41:, 0]
y_test = arrayList[41:, 1]

#visualize the data
plt.scatter(x_test, y_test, color='blue')
plt.show()

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean(y_true):
    return np.mean(y_true)
def variance(y_true):
    return np.var(y_true)

def covariance(x, y):
    return np.mean(x * y) - np.mean(x) * np.mean(y)

def covariance_matrix(x, y):
    return np.cov(x, y)

def simple_linear_regression(x, y):
    # Calculate the mean of x and y
    x_mean = mean(x)
    y_mean = mean(y)

    # Calculate the variance of x
    x_variance = variance(x)

    # Calculate the covariance of x and y
    cov = covariance(x, y)

    # Calculate the value of m
    m = cov / x_variance

    # Calculate the value of c
    c = y_mean - m * x_mean

    return m, c

def evaluate_model(x, y, m, c):
    # Calculate the predicted values
    y_pred = m * x + c

    # Calculate the mean squared error
    mse = MSE(y, y_pred)

    return mse

#visualize the model   
m, c = simple_linear_regression(x_test, y_test)
y_pred = m * x_test + c
plt.xlabel("X", style='italic')
plt.ylabel("Y", style='italic')
plt.scatter(x_test, y_test, color='blue')
plt.plot(x_test, y_test, color='red')
plt.show()