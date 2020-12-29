############################################################
# Overview
#
# Implements simulated data set generation from 
# Flexible Metric Nearest Neighbor Classification 
# by Friedman (1994)
############################################################
import numpy as np

def friedman_1(N = 200, P = 10):
    
    N1 = int(np.floor(N /2))
    N2 = int(N - N1)
    
    X = np.zeros([N, P])
    Y = np.zeros([N, 1])
    
    X[:N1, :] = np.random.normal(0, 1, [N1, P])
    Y[:N1, 0] = 0
    
    dims = np.arange(0, P)
    mu = (dims + 1)**.5 / 2
    sd = 1 / (dims + 1)**.5
    X[N1:, :] = np.random.normal(mu, sd, [N2, P])
    Y[N1:, 0] = 1
    
    return X, Y

def friedman_2(N = 200, P = 10):
    
    N1 = int(np.floor(N /2))
    N2 = int(N - N1)
    
    X = np.zeros([N, P])
    Y = np.zeros([N, 1])
    
    X[:N1, :] = np.random.normal(0, 1, [N1, P])
    Y[:N1, 0] = 0

    dims = np.arange(0, P)
    mu = (P - (dims + 1) + 1)**.5 / 2
    sd = 1 / (dims + 1)**.5
    X[N1:, dims] = np.random.normal(mu, sd, [N2, P])
    Y[N1:, 0] = 1
    
    return X, Y

def friedman_3(N = 200, P = 10):
    X = np.zeros([N, P])
    Y = np.zeros([N, 1])
    
    X[:, :] = np.random.normal(0, 1, [N, P])

    temp = X **2
    dims = np.arange(0, P) + 1
    temp = temp / dims
    temp = np.sum(temp, axis = 1)
    
    Y[:, 0] = np.where(temp <= 2.5, 0, 1)
    
    return X, Y

def friedman_4(N = 200, P = 10):
    X = np.zeros([N, P])
    Y = np.zeros([N, 1])
    
    X[:, :] = np.random.normal(0, 1, [N, P])
    
    temp = X **2
    temp = np.sum(temp, axis = 1)
    
    Y[:, 0] = np.where(temp <= 9.8, 0, 1)
    
    return X, Y

def friedman_5(N = 200, P = 10):
    X = np.zeros([N, P])
    Y = np.zeros([N, 1])
    
    X[:, :] = np.random.normal(0, 1, [N, P])
    
    temp = np.sum(X, axis = 1)
    
    Y[:, 0] = np.where(temp <= 0, 0, 1)
    
    return X, Y

def waveform(N = 200):
    
    P = 4
    X = np.zeros([N, P])
    Y = np.zeros([N, 1])
    
    def h1(i):
        temp = 6 - np.abs(i + 1 -7)
        return np.where(temp > 0, temp, 0)
    
    def h2(i):
        temp = 6 - np.abs(i + 1 -15)
        return np.where(temp > 0, temp, 0)
    
    def h3(i):
        temp = 6 - np.abs(i + 1 -11)
        return np.where(temp > 0, temp, 0)
    
    N1 = int(np.floor(N / 3))
    N2 = int(np.floor(N / 3))
    N3 = N - N1 - N2
    
    dims = np.arange(0, P)
    
    # class 0
    for i in np.arange(0, N1):
        for dim in dims:
            u = np.random.uniform(0, 1, 1)
            X[i, dim] = u * h1(dim) + (1 - u) * h2(dim) + np.random.normal(0, 1, 1)
            Y[i, 0] = 0
            
    # class 1
    for i in np.arange(N1, N1 + N2):
        for dim in dims:
            u = np.random.uniform(0, 1, 1)
            X[i, dim] = u * h1(dim) + (1 - u) * h3(dim) + np.random.normal(0, 1, 1)
            Y[i, 0] = 1
            
    # class 2
    for i in np.arange(N1 + N2, N1 + N2 + N3):
        for dim in dims:
            u = np.random.uniform(0, 1, 1)
            X[i, dim] = u * h2(dim) + (1 - u) * h3(dim) + np.random.normal(0, 1, 1)
            Y[i, 0] = 2
    
    return X, Y


