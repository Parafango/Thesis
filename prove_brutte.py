import numpy as np
from itertools import permutations
import pandas as pd
from scipy.interpolate import NearestNDInterpolator
import time
import matplotlib.pyplot as plt
import random

print('s')
print(random.gauss(0, 0.1 * 13))

'''
sigma_lin = np.linspace(33.2, 90, 8)
sigmac = np.mean([33.2, 90])


sigma_par = np.linspace(np.square(33.2), np.square(sigmac), 4)
sigma_par = np.sqrt(sigma_par)
sigma_par = np.concatenate((sigma_par, np.add(sigmac,np.abs(np.add(np.flip(sigma_par[0:len(sigma_par)]), -sigmac)))))
print(sigma_par)
y = np.zeros(np.shape(sigma_lin))

plt.plot(sigma_lin, y, 'r.')
plt.plot(sigma_par, y, 'b.')
plt.show()
'''