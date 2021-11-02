import numpy as np

#from ..processing.matching_tools_frequency_subpixel import phase_jac

def squared_difference(A,B):
    """ efficient computation of the squared difference

    Parameters
    ----------
    A : np.array, size=(m,n)
        data array
    B : np.array, size=(m,n)
        data array

    Returns
    -------
    sq_diff : float
        sum of squared difference.

    Notes
    -----
    .. math :: \Sigma{[\mathbf{A}-\mathbf{B}]^2}
    """
    diff = A-B
    sq_diff = np.einsum('ijk,ijk->',diff,diff)
    return sq_diff

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

def secant(A, y, J, params, n_iters=5, print_diagnostics=False):
    """
    also known as Boyden's method

    Parameters
    ----------
    A : np.array, size=(m,2), dtype=float
        design matrix
    y : np.array, size=(m,1), dtype=float
        phase angle vector
    J : np.array, size=(m,2), dtype=float
        Jacobian matrix
    params : np.array
        initial estimate
    n_iters : integer, optional
        amount of maximum iterations that need to be excuted, the default is 5.
    print_diagnostics : boolean
        print the solution at each iteration

    Returns
    -------
    params : np.array, dtype=float
        final estimate
    history : np.array, size=(n_iters)
        evolution of the cost per estimation in the iteration.

    """
    history = np.zeros((n_iters))
    
    if len(params)==1: # one dimensional problem
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
    else: # multi-variate case
        x0 = params.copy()-.1
        x1 = params.copy()
        
        for i in range(n_iters):
            fx0 = np.squeeze(A @ x0) - y
            fx1 = np.squeeze(A @ x1) - y
            
            delta_x = x1 - x0
            delta_f = fx1 - fx0
            # estimate Jacobian
            J_new = J + \
                np.outer( (delta_f - (J @ delta_x)), delta_x) / \
                np.dot(delta_x,delta_x)
            
            # estimate new parameter set
            dx = np.linalg.lstsq(J_new, -fx1, rcond=None)[0]
            x2 = x1 + dx
            
            if print_diagnostics:
                print('di:{:+.4f}'.format(x2[0])+' dj:{:+.4f}'.format(x2[1]))
            # update
            x0,x1,J = x1.copy(), x2.copy(), J_new.copy()
            history[i] = compute_cost(A, y, x1)
        params = x1
        
    if (history[0]<history[-1]) and print_diagnostics:
        print('no convergence')
    return params, history    

    