import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from Clean import train_x,dev_x,test_x,train_y,dev_y,test_y



def standardize(csv):
    #csv = csv - np.mean(csv, axis=1, keepdims=True)
    #csv /= np.std(csv, axis=1, keepdims=True)
    csv = csv / (np.max(csv, axis = 1, keepdims=True))
    return csv

def random_weights(csv):
    shape = csv.shape
    return np.random.randn(shape[1], 1), random.random()

def calc_cost(y, y_hat):
    squared = np.square(y_hat - y)
    const = 2 * squared.shape[0]
    return np.sum(squared) / const

def calc_derivative(y, y_hat, X):
    diff = y_hat - y
    dW = (np.sum(np.dot(diff.T, X), axis=0, keepdims=True) / X.shape[0]).T
    db = np.sum(diff) / X.shape[0]
    return dW, db

def update_derivative(w, b, dw, db, alpha):
    w -= alpha * dw
    b -= alpha * db
    return w, b

def pred_y_hat(w, b, X):
    return np.dot(X, w) + b

def run_regression(epoch, alpha, X, y):
    w, b = random_weights(X)
    for i in range(epoch):
        y_hat = pred_y_hat(w, b, X)
        dw, db = calc_derivative(y, y_hat, X)
        w, b = update_derivative(w, b, dw, db, alpha)
    return w, b

def tune_alpha(epoch, train_x, train_y, dev_x, dev_y):
    alpha = [0.000001, 0.000003, 0.00001, 0.00003]
    costs = list()
    params = list()
    for a in alpha:
        w, b = run_regression(epoch, a, train_x, train_y)
        y_hat = pred_y_hat(w, b, dev_x)
        cost = calc_cost(y_hat,dev_y)
        costs.append(cost)
        params.append((w,b))
    return costs.index(min(costs)),costs, params


