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

def map_features(X, degree):
    out = X
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            for k in range(degree+1):
                # print(i,"^",k, j, "^", degree - k)
                out = np.append(out, np.reshape(np.power(X[:,i], k) * np.power(X[:,j], degree - k), (-1,1)), axis = 1)
    return out
# out = np.append(X, np.reshape(X[:,0] * X[:,1], (-1,1)), axis = 1)

if __name__ == '__main__':
    all_data = load_data()
    train_data , test_data = partition_data(all_data)
    m = train_data.shape[0]

    X = train_data[:, :-1]
    Y = train_data[:, -1]
    X_test = test_data[:, :-1]
    Y_test = test_data[:, -1]
    Y = np.reshape(Y, (-1,1))
    # X = np.reshape(X, (-1,1))
    # X_test = np.reshape(X_test, (-1,1))


    # X_norm, X_mu, X_sigma = feature_normalization(X)


    X = map_features(X, 10)
    X_test = map_features(X_test, 10)
    print(X.shape)
    n = X.shape[1]
    w, b = initialize_with_zeros(n)

    X_norm, X_mu, X_sigma = feature_normalization(X)
    cost_save, w, b, dw, db = model(X_norm, Y, w, b, 5000, 0.8)

    Y_prediction_test = predict(X_test, w, b, X_mu, X_sigma)
    Y_prediction_train = predict(X, w, b, X_mu, X_sigma)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    print("cost:", cost_save[-1])
    print(X.shape)

    # plt.plot(X, Y, 'o')
    # plt.show()
    plt.plot(cost_save)
    plt.show()
