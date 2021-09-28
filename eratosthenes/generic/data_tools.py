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

def secant(A, y, J, params, n_iters=5):
    """
    also known as Boyden's method

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    J : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.
    n_iters : TYPE, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    params : TYPE
        DESCRIPTION.
    history : TYPE
        DESCRIPTION.

    """
    history = np.zeros((n_iters))
    
    x0 = params.copy()-.1
    x1 = params.copy()
    for i in range(n_iters):
        fx0 = np.squeeze(A @ x0) - y
        fx1 = np.squeeze(A @ x1) - y

        x2 = x0 - ((x1-x0)*np.sum(fx0))/( np.sum(fx1 - fx0) )
        
        # update
        x0 = x1.copy()
        x1 = x2.copy()
        history[i] = compute_cost(A, y, x1)
    if history[0]<history[-1]:
        print('no convergence')
    return params, history    
    