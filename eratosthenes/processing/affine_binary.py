import numpy as np
from scipy.optimize import fsolve

from PIL import Image

img = Image.open('template01.png')
I = np.array(img)
B1 = np.double(I[:,:,1])/255

img = Image.open('observation01.png')
I = np.array(img)
B2 = np.double(I[:,:,1])/255
del img, I

# preparation
pT = np.sum(B1) # Lebesgue integral
pO = np.sum(B2)

Jac = pO/pT # Jacobian

x = np.linspace(0,B1.shape[1]-1,B1.shape[1])
y = np.linspace(0,B1.shape[0]-1,B1.shape[0])
X1, Y1 = np.meshgrid(x,y)
del x, y

# calculating moments of the template
x11 = Jac* np.sum(X1    * B1)
x12 = Jac* np.sum(X1**2 * B1)
x13 = Jac* np.sum(X1**3 * B1)

x21 = Jac* np.sum(Y1    * B1)
x22 = Jac* np.sum(Y1**2 * B1)
x23 = Jac* np.sum(Y1**3 * B1)
del X1, Y1

x = np.linspace(0,B2.shape[1]-1,B2.shape[1])
y = np.linspace(0,B2.shape[0]-1,B2.shape[0])
X2, Y2 = np.meshgrid(x,y)
del x, y

# calculating moments of the observation
y1   = np.sum(X2       * B2)
y12  = np.sum(X2**2    * B2)
y13  = np.sum(X2**3    * B2)
y12y2= np.sum(X2**2*Y2 * B2)
y2   = np.sum(Y2       * B2)
y22  = np.sum(Y2**2    * B2)
y23  = np.sum(Y2**3    * B2)
y1y22= np.sum(X2*Y2**2 * B2)
y1y2 = np.sum(X2*Y2    * B2)
del X2, Y2

# estimation

mu = pO

def func1(x):
    q11, q12, q13 = x
    return [mu*q11 + y1*q12 + y2*q13 - x11,
            mu*q11**2 + y12*q12**2 + y22*q13**2 + 2*y1*q11*q12 + 2*y2*q11*q13 + 2*y1y2*q12*q13 - x12,
            mu*q11**3 + y13*q12**3 + y23*q13**3 + 3*y1*q11**2*q12 + 3*y2*q11**2*q13 + 3*y12*q12**2*q11 + 3*y12y2*q12**2*q13 + 3*y22*q11*q13**2 + 3*y1y22*q12*q13**2 + 6*y1y2*q11*q12*q13 - x13]

Q11, Q12, Q13 = fsolve(func1, (1.0, 1.0, 1.0))
# test for complex solutions, which should be excluded

def func2(x):
    q21, q22, q23 = x
    return [mu*q21 + y1*q22 + y2*q23 - x12,
            mu*q21**2 + y12*q22**2 + y22*q23**2 + 2*y1*q21*q22 + 2*y2*q21*q23 + 2*y1y2*q22*q23 - x22,
            mu*q21**3 + y13*q22**3 + y23*q23**3 + 3*y1*q21**2*q22 + 3*y2*q21**2*q23 + 3*y12*q22**2*q21 + 3*y12y2*q22**2*q23 + 3*y22*q21*q23**2 + 3*y1y22*q22*q23**2 + 6*y1y2*q21*q22*q23 - x13]

Q21, Q22, Q23 = fsolve(func2, (1.0, 1.0, 1.0))
# test for complex solutions, which should be excluded

Q = np.array([[Q12, Q13, Q11], [Q22, Q23, Q21]])

# search for the correct combination
#dMin = float('inf')
#for k in range(len(Q11)):
#    for l in range(len(Q21)):
#        tJac = # determinant






