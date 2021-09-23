import numpy as np

def compute_cost(A, y, params):
    n_samples = len(y)
    hypothesis = np.squeeze(A @ params)
    return (1/(2*n_samples))*np.sum(np.abs(hypothesis-y))

def gradient_descent(A, y, params, learning_rate=0.01, n_iters=100):
    n_samples, history = y.shape[0], np.zeros((n_iters))

    for i in range(n_iters):
        err = np.squeeze(A @ params) - y
        
        # err = (1 / (1 + err**2))*err
        
        # IN = np.abs(err)<=np.quantile(np.abs(err), quantile)
        # err = err[IN]
        # params = params - (learning_rate/n_samples) * A[IN,:].T @ err
        params = params - (learning_rate/n_samples) * A.T @ err
        history[i] = compute_cost(A, y, params)
    if history[0]<history[-1]:
        print('no convergence')
    return params, history