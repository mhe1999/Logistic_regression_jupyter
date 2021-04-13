import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
# from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_data(path = "LogisticRegression_Diabetes.csv"):
    data = pd.read_csv(path)
    dataset = data.values
    return dataset

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w, b

def sigomid(z):
    return 1/(1 + np.exp(-z))

def hyp(X, w, b):
    # print("X:", X.shape)
    # print("w:",w.shape)
    # print(":",w.shape)
    t = np.dot(X, w)
    # print("t:",t.shape)
    # print("t+b", (t+b).shape)
    h = sigomid(t + b)
    # print("h:",h.shape)

    return h

def compute_cost(X, y, w, b):
    epsilon = 1e-5
    m = X.shape[0]
    h = hyp(X, w, b)
    # loss = (-y * np.log(h) - (1-y) * np.log(1-h))
    # cost = 1/m * np.sum(loss)
    # print('y.t', y.T.shape)
    # print('np.log(h)',np.log(h).shape)
    cost = -1/m * (np.dot(y.T, np.log(h + epsilon)) + np.dot((1-y).T, np.log(1-h + epsilon)))
    return cost

def cal_gradients(X, y, w, b):
    m = X.shape[0]
    h = hyp(X, w, b)
    dw = 1/m * np.dot(X.T, (h-y))
    db = 1/m * np.sum(h-y)
    return dw, db

def update_parameters(dw, db, w, b, alpha):
    w = w - alpha * dw
    b = b - alpha * db
    return w, b

def feature_normalization(X):
    X_norm = X;
    mu = np.zeros((1, X.shape[1]));
    sigma = np.zeros((1, X.shape[1]));
    for i in range(X.shape[1]):
        mu[0,i] = np.mean(X[:, i])
        sigma[0,i] = np.std(X[:, i])
    # print("sigma:", sigma)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def model(X, y, w, b, itteration = 1500, alpha = 0.01):
    m = X.shape[0]
    # theta = inital_theta
    cost_save = np.array([])
    for i in range(itteration):
        dw, db = cal_gradients(X, y, w, b)
        w, b = update_parameters(dw, db, w, b, alpha)
        cost = compute_cost(X, y, w, b)
        # print(i, cost)
        if i % 100 == 0:
            print(i, cost)
            # pass
        cost_save= np.append(cost_save, cost)
    return cost_save, w, b, dw, db

def predict(X, w, b, mu, sigma):
    m = X.shape[0]
    # print(X.shape)
    # print(theta.shape)
    # print(mu.shape)
    # print(sigma.shape)
    X_norm = (X - mu) / sigma
    # X_norm = np.append(np.zeros((m,1)) + 1, X_norm, axis = 1)
    yhat = sigomid(np.dot(X_norm, w) + b)
    Y_prediction = np.round(yhat)
    return Y_prediction

def partition_data(dataset,train_percent = 0.8, test_percent = 0.2):
    m = dataset.shape[0]
    train_number = math.floor(train_percent * m)
    test_number = math.floor(test_percent * m)
    # print(type(dataset))
    train_data = dataset[0:train_number, :]
    test_data = dataset[train_number:, :]


    return train_data, test_data

if __name__ == '__main__':
    all_data = load_data()
    train_data , test_data = partition_data(all_data)
    X = train_data[:, :-1]
    Y = train_data[:, -1]
    X_test = test_data[:, :-1]
    Y_test = test_data[:, -1]
    # print(X.shape)
    m = X.shape[0]
    n = X.shape[1]
    X_norm, X_mu, X_sigma = feature_normalization(X)
    # X_norm = np.append(np.zeros((m,1)) + 1, X_norm, axis = 1)

    Y = np.reshape(Y, (-1,1))
    # theta = np.random.rand(X_norm.shape[1])
    # theta = np.reshape(theta, (-1,1))
    w, b = initialize_with_zeros(n)
    # print("x:", X_norm)
    # print("y:", y)
    # print("w:",w)
    # print("b:", b)
    # theta = np.loadtxt('data.csv', delimiter=',')
    # theta = np.reshape(theta, (-1,1))
    # c,t = model(X_norm,y,theta, 1000, 1)
    # np.savetxt('data.csv', t, delimiter=',')
    # w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1,3],[2,4], [-1,-3.2]]), np.array([[1],[0],[1]])
    # print(X.shape)
    # print(w.shape)
    cost_save, w, b, dw, db = model(X_norm, Y, w, b, 100, 90)
    # print(predict(X,w,b, X_mu, X_sigma))
    # cost = compute_cost(X,Y, w, b)
    # dw, db = cal_gradients(X, Y, w, b)
    # print("w", w)
    # print("b", b)
    # print("dw", dw)
    # print("db", db)
    # print("cost:", cost)
    # print(t)
    # red = np.where(all_data[:, -1]==1)
    # blue = np.where(all_data[:, -1]==0)
    # print(red)
    # print('\n\n\n')
    # plt.plot(X[red], 'ro')
    # plt.plot(X[blue], 'bo')
    # plt.plot(X,y, 'bo')

    # plt.show()
    # yhat = predict(X[0:1,0:-1], theta, X_mu, X_sigma)
    # print(yhat)


    # print(X.shape)
    # print(y.shape)
    # print(theta.shape)
    # print("x:", X)
    # print("y:", y)
    # print("theta:", theta)
    # print(cost_funtion(X, y, theta).shape)
    # print(cost_funtion(X, y, theta))
    # print(data.shape[1])
