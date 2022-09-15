import numpy as np
import matplotlib.pyplot as plt

from dhdt.generic.filtering_statistical import \
    online_steady_state, online_t_test

N = 100
c = 1
x = np.linspace(0,7,N)
# create exponential decay
y = np.exp(-c*x)
# add artificial noise
y += np.random.normal(loc=0.0, scale=0.05, size=N)

#plt.figure(), plt.plot(x,y)
print('.')

# parameters
lambda_1, lambda_2 , lambda_3 = 0.2, 0.1, 0.1
R_thres = 1.44

R_i, T_i = np.zeros_like(y), np.zeros_like(y)
X,v,d = None, None, None
for idx,val in enumerate(y):
    y_ol = y[:idx+1] # simulate online
    R_i[idx],X,v,d = online_steady_state(y_ol,X,v,d)
    T_i[idx] = online_t_test(y_ol)

print('.')
plt.figure(), plt.plot(x,R_i)
