import numpy as np

def average(X, ws = 1):
    ret = []
    for i in range(ws):
        ret.append(X[i])
    for i in range(ws, X.shape[0] - ws):
        ret.append(np.mean(X[i - ws : (i + ws + 1)], axis = 0))
    for i in range(X.shape[0] - ws, X.shape[0]):
        ret.append(X[i])
    ret = np.array(ret)
    return ret

def get_weights(ws):
    weights = []
    for i in range(1, ws + 2): weights.append(i)
    for i in range(ws, 0, -1): weights.append(i)
    # print(len(weights))
    return weights

def weight_average(X, ws = 1):
    ret = []
    for i in range(ws):
        ret.append(X[i])
    for i in range(ws, X.shape[0] - ws):
        weights = get_weights(ws)
        ret.append(np.average(X[i - ws : (i + ws + 1)], axis = 0, weights = weights))
    for i in range(X.shape[0] - ws, X.shape[0]):
        ret.append(X[i])
    ret = np.array(ret)
    return ret

def concatenate(X, ws=1):
    ret = []
    for i in range(ws):
        x = list(X[i])
        # print(np.array(x * (2 * ws + 1)).shape)
        ret.append(np.array(x * (2 * ws + 1)))
    for i in range(ws, X.shape[0] - ws):
        ret.append(np.concatenate(X[i - ws: (i + ws + 1)]))
    for i in range(X.shape[0] - ws, X.shape[0]):
        x = list(X[i])
        ret.append(np.array(x * (2 * ws + 1)))
    ret = np.array(ret)
    # print(ret.shape)
    return ret

def left_concatenate(X, ws=1):
    ret = []
    for i in range(ws):
        x = list(X[i])
        # print(np.array(x * (2 * ws + 1)).shape)
        ret.append(np.array(x * (ws + 1)))
    for i in range(ws, X.shape[0]):
        ret.append(np.concatenate(X[i - ws: (i + 1)]))
    ret = np.array(ret)
    # print(ret.shape)
    return ret

def right_concatenate(X, ws=1):
    ret = []
    for i in range(0, X.shape[0] - ws):
        ret.append(np.concatenate(X[i: (i + ws + 1)]))
    for i in range(X.shape[0] - ws, X.shape[0]):
        x = list(X[i])
        ret.append(np.array(x * (ws + 1)))
    ret = np.array(ret)
    # print(ret.shape)
    return ret
