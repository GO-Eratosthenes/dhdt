import os
import numpy as np
import matplotlib.pyplot as plt

fpath = '/Users/Alten005/Documents/Papers/Atmosphere'
fname = 'atmospheric-concentration-of-carbon-dioxide-5.csv'

ffull = os.path.join(fpath, fname)

my_co2 = np.genfromtxt(ffull, delimiter=',',skip_header=1)
co2 = my_co2[40:87,:-1]

pf = np.polyfit(co2[:,0],co2[:,1],2)
p = np.poly1d(pf)
xp = np.linspace(1950, 2025, 100)

p_co2 = np.poly1d(np.array([0.012786034930234986,
                            -49.31617858270089,
                            47857.733173381115]))

# triangular wave

plt.plot(xp, p(xp))
#plt.plot(xp, p_round(xp))
plt.scatter(co2[:,0],co2[:,1])
plt.show()

print('read data')