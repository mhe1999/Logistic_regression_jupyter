import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
# from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_data(path = "LogisticRegression_Diabetes.csv"):
    data = pd.read_csv(path)
    dataset = data.values
    return dataset

def sigomid(z):
    return 1/(1 + np.exp(-z))

def hyp(X, theta):
    t = np.dot(X, theta)
    h = sigomid(t)
    return h

def compute_cost(X, y, theta):
    m = X.shape[0]
    h = hyp(X, theta)
    loss = (-y * np.log(h) - (1-y) * np.log(1-h))
    cost = 1/m * np.sum(loss)
    return cost

def cal_gradients(X, y, theta):
    m = X.shape[0]
    h = hyp(X, theta)
    gradients = 1/m * np.dot(X.T, (h-y))
    return gradients

if __name__ == '__main__':
    all_data = load_data()
    X = all_data[0:1, :-1]
    y = all_data[0:1, -1]
    m = X.shape[0]

    X = np.append(np.zeros((m,1)) + 1, X, axis = 1)
    y = np.reshape(y, (-1,1))
    theta = np.zeros((X.shape[1]))
    theta = np.reshape(theta, (-1,1))
    # print(X.shape)
    # print(y.shape)
    # print(theta.shape)
    # print("x:", X)
    # print("y:", y)
    # print("theta:", theta)
    # print(cost_funtion(X, y, theta).shape)
    # print(cost_funtion(X, y, theta))
    # print(data.shape[1])
    print(cal_gradients(X, y, theta))
