import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

def gauss(x, a, b, c):
    return a * np.exp(-(x - b) ** 2.0 / (2 * c ** 2))

grid_height = 185
grid_width = 185
hub_height = 119
n_rows = 10
n_cols = 10
Vm = 13
shear_exp = 0.12
a = 3
b = 0
c = 45
gauss_par = (a, b, c)
y_c = 0
z_c = 123

y_arr = np.linspace(-grid_width / 2, grid_width / 2, n_cols)
z_arr = np.linspace(hub_height - grid_height / 2, hub_height + grid_height / 2, n_rows)

Y, Z = np.meshgrid(y_arr, z_arr)

U_shear = np.zeros(np.shape(Y))
U_gauss = np.zeros(np.shape(Y))

for i in range(0, n_rows):
    U_shear[i, :] = Vm * np.power(z_arr[i]/hub_height, shear_exp)
    for j in range(0, n_cols):
        r = np.sqrt(np.add(np.square(np.add(y_arr[j], -y_c)), np.square(np.add(z_arr[i], -z_c))))
        U_gauss[i, j] = gauss(r, *gauss_par)

U = np.add(U_shear, -U_gauss)

time_series_file = open('C:/Users/randr/Desktop/TimeSeriesBlank.txt', 'a')

for i in range(0, n_rows):
    for j in range(0, n_cols):
        time_series_file.write('\n\t{}\t\t{}'.format(np.round(Y[i, j], 3), np.round(Z[i, j], 3)))

time_series_file.write('\n--------Time Series-------------------------------------------------------------\n')
time_series_file.write('Elapsed Time\t')

for i in range(0, n_rows):
    for j in range(0, n_cols):
        time_series_file.write('Point{:03d}u\t'.format(i * 10 + j + 1))

time_series_file.write('\n')
time_series_file.write('(s)\t\t')
for i in range(0, n_rows):
    for j in range(0, n_cols):
        time_series_file.write('(m/s)\t\t')

time_series_file.write('\n')

t0 = 0.000
dt = 0.05
tf = 600
t_len = int(tf / dt + 1)

for i in range(0, t_len):
    t = t0 + i * dt
    time_series_file.write('{:.3f}\t\t'.format(t))
    for i in range(0, n_rows):
        for j in range(0, n_cols):
            time_series_file.write('{}\t\t'.format(np.round(U[i, j], 3)))

    time_series_file.write('\n')

time_series_file.close()

'''
fig, axes = plt.subplots(3,1, subplot_kw={"projection": "3d"})

surf = axes[0].plot_surface(Y, Z, U_shear, cmap=cm.Spectral,
                       linewidth=0, antialiased=False)


surf2 = axes[1].plot_surface(Y, Z, U_gauss, cmap=cm.Spectral,
                             linewidth=0, antialiased = False)

surf3 = axes[2].plot_surface(Y, Z, U, cmap=cm.Spectral,
                             linewidth=0, antialiased = False)

fig.colorbar(surf3)
axes[0].title.set_text('Shear')
axes[1].title.set_text('Gaussian wake')
axes[1].title.set_text('Total')

plt.show()
'''