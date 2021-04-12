import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
# from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_data(path = "linearRegression_carPrice.csv"):
    data = pd.read_csv(path)
    dataset = data.values
    return dataset

def sigomid(z):
    return 1/(1 + np.exp(-z))

if __name__ == '__main__':

    a = np.array([1,2,3])
    print(sigomid(a))
