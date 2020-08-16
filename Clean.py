import numpy as np
csv = np.genfromtxt('Summary of Weather.csv', delimiter=",", usecols=np.arange(0,8))
import math
def clean(csv):
    y = csv[:, 4]
    csv = np.delete(csv, [0,1,3,4,7], axis=1)
    csv = np.delete(csv, 0, axis=0)
    list_to_delete = []
    for i in range(csv.shape[0]):
        if True in np.isnan(csv[i, :]):
            list_to_delete.append(i)
    csv = np.delete(csv, list_to_delete, axis=0)
    y = np.delete(y, 0, axis=0)
    y = np.delete(y,list_to_delete,axis=0)
    return csv,y

def split(csv,y):
    train = math.floor(csv.shape[0]*0.6)
    dev = math.floor((csv.shape[0]-train)/2)
    test = csv.shape[0]-train-dev
    train_x = csv[0:train]
    train_y = y[0:train]
    dev_x = csv[train:train+dev]
    test_x = csv[train+dev+test:csv.shape[0]]
    train_y = y[0:train]
    dev_y = y[train:train + dev]
    test_y = y[train + dev + test:csv.shape[0]]
    return train_x,dev_x,test_x,train_y,dev_y,test_y

    return train_x,train_y
"""
def clean(csv):
    csv = np.delete(csv, [7, 8, 9, 13, 14, 18], axis=1)
    csv = np.delete(csv, 0, axis=0)
    list_to_delete = []
    for i in range(csv.shape[0]):
        if True in np.isnan(csv[i, :]):
            list_to_delete.append(i)
    csv = np.delete(csv, list_to_delete, axis=0)

    y = csv[:, 2]
    csv = np.delete(csv, 2, axis=1)
    return csv, y
"""
csv,y = clean(csv)
train_x, dev_x, test_x, train_y, dev_y, test_y  = split(csv,y)



