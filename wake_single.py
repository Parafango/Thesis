import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import time
import numpy as np
import pandas as pd
import math
from scipy import stats
from scipy.optimize import curve_fit
from floris.tools import FlorisInterface
from floris.tools.visualization import plot_rotor_values
from floris.tools.visualization import visualize_cut_plane
import floris.tools.visualization as wakeviz
from floris.tools.cut_plane import  get_plane_from_flow_data

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------Functions for gauss parameters calculation------------------------------------------
def flatten_wind_field(Z, U, alfa, Vref, href):
    z_arr = Z[:,0]
    U_new = np.zeros(np.shape(U))
    for i, z in enumerate(z_arr):
        ws = np.ones(np.shape(U)[1])
        shear = Vref * pow((z/href), alfa)
        ws = np.multiply(ws, shear)
        U_new[i,:] = U[i,:] - ws

    return U_new


def get_U_squared(U2s, Y, Z):
    toll = -1e-1
    counter = 0
    z_arr = Z[:, 0]
    y_arr = Y[0, :]
    flag = 0

    for i in range(U2s.shape[0]):
        diff = toll - min(U2s[i, :])
        if diff > 0:
            counter = counter + 1

        if counter == 1:
            zmin = z_arr[i]

        if diff < 0 and counter > 1 and flag == 0:
            flag = 1
            zmax = z_arr[i - 1]

    counter = 0
    flag = 0

    for i in range(U2s.shape[1]):
        diff = toll - min(U2s[:, i])

        if diff > 0:
            counter = counter + 1

        if counter == 1:
            ymin = y_arr[i]

        if diff < 0 and counter > 1 and flag == 0:
            flag = 1
            ymax = y_arr[i - 1]


    limits = [[ymin, ymax], [zmin, zmax]]

    n = U2s.shape[0]  # let's assume n is odd for now

    u_min = np.amin(U2s)

    idx_min = [U2s == u_min]

    y_c = Y[idx_min[0]]
    z_c = Z[idx_min[0]]

    # impose symmetry with respect to offset (yc !=0 generally)
    if abs(ymin - y_c) > (ymax - y_c):
        ry = abs(ymin - y_c)
    else:
        ry = abs(ymax - y_c)

    imin = np.where(abs(y_arr + ry - y_c) < 1e-4)[0]
    imax = np.where(abs(y_arr - ry - y_c) < 1e-4)[0]
    dim_i = imax[0] - imin[0] + 1
    jmin = np.where(abs(z_arr) < 1e-2)[0]
    jmax = np.where(z_arr == zmax)[0]
    dim_j = jmax[0] - jmin[0] + 1

    U_squared = U2s[jmin[0]:(jmax[0] + 1), imin[0]:(imax[0] + 1)]

    y_squared = y_arr[imin[0]:imax[0] + 1]
    z_squared = z_arr[jmin[0]:jmax[0] + 1]

    Y_squared, Z_squared = np.meshgrid(y_squared, z_squared)

    return U_squared, Y_squared, Z_squared, y_c, z_c

def get_U_squared_no_z0(U2s, Y, Z, fixed_grid=False, toll=1e-1):
    u_min = np.amin(U2s)
    idx_min = [U2s == u_min]
    y_c = Y[idx_min[0]]
    z_c = Z[idx_min[0]]
    z_arr = Z[:, 0]
    y_arr = Y[0, :]

    if fixed_grid:
        #create a 65x50 grid centered in yc, zc
        z_squared = z_arr[0:50]
        idx_yc = np.where(y_arr == y_c)[0][0]
        y_squared = y_arr[idx_yc-32:idx_yc+33]
        U_squared = U2s[0:50, idx_yc-32:idx_yc+33]

    else:
        idx_rect = [U2s <= -toll]
        y_rect = np.unique(Y[idx_rect[0]])
        z_rect = np.unique(Z[idx_rect[0]])
        ymin = y_rect[0]
        ymax = y_rect[-1]
        zmin = z_rect[1]
        zmax = z_rect[-1]

        # impose symmetry with respect to center of the wake (yc !=0 generally)
        if abs(ymin - y_c) > (ymax - y_c):
            ry = abs(ymin - y_c)
        else:
            ry = abs(ymax - y_c)


        if abs(zmin - z_c) > (zmax - z_c):
            rz = abs(zmin - z_c)
        else:
            rz = abs(zmax - z_c)

        imin = np.where(abs(y_arr - (y_c - ry)) < 1e-4)[0]
        imax = np.where(abs(y_arr - (y_c + ry)) < 1e-4)[0]
        jmin = [1] #always start from z>0
        jmax = np.where(abs(z_arr - (z_c + rz)) < 1e-4)[0]

        U_squared = U2s[jmin[0]:(jmax[0] + 1), imin[0]:(imax[0] + 1)]

        y_squared = y_arr[imin[0]:imax[0] + 1]
        z_squared = z_arr[jmin[0]:jmax[0] + 1]

    Y_squared, Z_squared = np.meshgrid(y_squared, z_squared)

    return U_squared, Y_squared, Z_squared, y_c, z_c

def gauss(x, a, b, c):
    return a * np.exp(-(x - b) ** 2.0 / (2 * c ** 2))

def gauss_c(xy, A, x0, y0, sigma):
    return A * np.exp(-0.5*(((xy[0]-x0)/(sigma)) ** 2.0 + ((xy[1]-y0)/(sigma)) ** 2.0))


def gauss_interp(Y_squared, Z_squared, y_c, href, U_squared):
    # in order to center the coordinates in the new reference system centered in yc, href
    Z_centered = np.add(Z_squared, -href)
    Y_centered = np.add(Y_squared, -y_c)

    R = np.sqrt(np.add(np.square(Y_centered), np.square(Z_centered)))
    R_flat = R.flatten()
    U_flat = U_squared.flatten()
    r_unq = np.unique(R_flat)

    data = {
        'radius': R_flat,
        'ws': U_flat
    }

    df = pd.DataFrame(data)

    df_sort = df.sort_values('radius')

    ws_unq = np.zeros(r_unq.shape)

    for i, r in enumerate(r_unq):
        idx = df_sort.index[df_sort['radius'] == r]
        ws_unq[i] = np.mean([df_sort['ws'][idx]])

    ws_unq = abs(ws_unq)

    r_arr = np.concatenate((np.flip(-r_unq, 0), r_unq))
    #to make the wind field symmetric with respect to r = 0, ws_unq is mirrored
    ws_arr = np.concatenate((np.flip(ws_unq, 0), ws_unq))

    popt, pcov = curve_fit(gauss, r_arr, ws_arr)

    perr = np.sqrt(np.diag(pcov))  # calculate stdv errors on parameters A, offset, sigma

    return popt, perr, r_arr, ws_arr

def gauss_interp_c(Y_squared, Z_squared, y_c, href, U_squared):
    # in order to center the coordinates in the new reference system centered in yc, href
    U_squared = np.abs(U_squared)
    Z_centered = np.add(Z_squared, -href)
    Y_centered = np.add(Y_squared, -y_c)

    yz_arr = np.vstack((Y_centered.ravel(), Z_centered.ravel()))
    u_arr = U_squared.ravel()

    popt, pcov = curve_fit(gauss_c, yz_arr, u_arr)

    perr = np.sqrt(np.diag(pcov))  # calculate stdv errors on parameters A, offset, sigma

    return popt, perr, yz_arr, u_arr

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------Farm and parameters definition------------------------------------------------------

fi = FlorisInterface('Configurations/gch.yaml')
fi.reinitialize(turbine_library_path='Configurations/turbine_library')
fi.reinitialize(turbine_type=['Baseline_10MW_0.yaml'])

#Definition of constants in code
D = fi.floris.farm.rotor_diameters[0]
href = fi.floris.farm.hub_heights[0]
alfa = fi.floris.flow_field.wind_shear
layout_x = [0., 7 * D]
layout_y = [0., 0.]

solver_settings = {
    "type": "turbine_grid",
    "turbine_grid_points": 3
}

fi.reinitialize(solver_settings=solver_settings, reference_wind_height=href)
fi.reinitialize(layout_x=layout_x, layout_y=layout_y)

#parameters for which the field must be calculated
imp_arr = np.array([-0.5, -0.25, 0, 0.25, 0.5]) #it is taken into account at the end
offset_arr = np.array([3, 4, 5, 6, 7])
TI_arr = np.array([0.02, 0.06, 0.1])
yaw_arr = np.array([-25, -15, 0, 15, 25])
Vm_arr = np.array([7, 10, 11.4, 12, 12.5, 13, 14])
der_arr = np.array([0, 2.5, 5, 10, 15])

l_i = len(imp_arr)
l_o = len(offset_arr)
l_y = len(yaw_arr)

#definition of gauss parameters storing matrix
gauss_parameters = np.zeros((np.shape(yaw_arr)[0] * np.shape(offset_arr)[0] * np.shape(imp_arr)[0], 10))

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------Loop begins over parameters---------------------------------------------------------
fi.reinitialize(wind_speeds = [Vm_arr[0]])
fi.reinitialize(turbulence_intensity = TI_arr[0])

der = int(der_arr[0])
der = str(der)
turbine_name = 'Baseline_10MW_' + der + '.yaml'


fi.reinitialize(turbine_type = [turbine_name])

Vref = fi.floris.flow_field.wind_speeds[0]

yaw_angles = np.zeros((1, 1, 2))
yaw_angles[0, 0, 0] = 25
fi.calculate_wake(yaw_angles=yaw_angles)

cross_plane = fi.calculate_cross_plane(
    y_resolution=51,
    z_resolution=51,
    downstream_dist=5 * D,
    yaw_angles=yaw_angles
)

df = cross_plane.df
y_grid = np.array(df['x1'])
z_grid = np.array(df['x2'])
u_grid = np.array(df['u'])

n = int(math.sqrt(u_grid.shape[0]))
Y = y_grid.reshape(n, n)
Z = z_grid.reshape(n, n)
U = u_grid.reshape(n, n)

U_new = flatten_wind_field(Z, U, alfa, Vref, href)
U_squared, Y_squared, Z_squared, y_c, z_c = get_U_squared_no_z0(U_new, Y, Z)
popt, perr, yz_arr, u_arr = gauss_interp_c(Y_squared, Z_squared, y_c, z_c, U_squared)

print(popt)

'''
start_time = time.time()

for i_y, yaw_angle in enumerate(yaw_arr):
    yaw_angles = np.zeros((1, 1, 1))
    yaw_angles[0, 0, 0] = yaw_angle
    fi.calculate_wake(yaw_angles=yaw_angles)

    for i_dist, distance in enumerate(offset_arr):
        cross_plane = fi.calculate_cross_plane(
            y_resolution=51,
            z_resolution=51,
            downstream_dist=distance * D,
            yaw_angles=yaw_angles
        )

        df = cross_plane.df
        y_grid = np.array(df['x1'])
        z_grid = np.array(df['x2'])
        u_grid = np.array(df['u'])

        n = int(math.sqrt(u_grid.shape[0]))
        Y = y_grid.reshape(n, n)
        Z = z_grid.reshape(n, n)
        U = u_grid.reshape(n, n)

        U_new = flatten_wind_field(Z, U, alfa, Vref, href)
        U_squared, Y_squared, Z_squared, y_c, z_c = get_U_squared(U_new, Y, Z)
        popt, perr, r_arr, ws_arr = gauss_interp(Y_squared, Z_squared, y_c, z_c, U_squared)

        A, mu, sigma = popt

        A = round(A, 3)
        sigma = round(sigma, 3)

        interp_values = np.array(gauss(r_arr, *popt))
        dev = np.abs(np.add(interp_values, -ws_arr))
        err_max = round(np.amax(dev), 3)
        err_mean = round(np.mean(dev), 3)
        stdv = round(np.std(dev), 3)

        for i_i, imp in enumerate(imp_arr):
            offset = round(y_c[0] - imp * D, 3)
            popti = np.array([yaw_angle, distance, imp, A, offset, sigma, Vref, err_max, err_mean, stdv])
            gauss_parameters[i_y * l_o * l_i + l_i * i_dist + i_i, :] = popti


everything = pd.DataFrame(gauss_parameters, columns=['yaw', 'distance', 'imp', 'peak', 'yc', 'sigma', 'Vm', 'e_max', 'e_mean', 'e_stdv'])


end_time = time.time()
time_elapsed = end_time-start_time

print(f'Process took {time_elapsed} seconds')

print('-----------------------------------------------------------------------------------------------------------------')
'''



'''
fig, axes = plt.subplots(2,1, subplot_kw={"projection": "3d"})

surf = axes[0].plot_surface(Y, Z, U, cmap=cm.Spectral,
                       linewidth=0, antialiased=False)


surf2 = axes[1].plot_surface(Y, Z, U_new, cmap=cm.Spectral,
                             linewidth=0, antialiased = False)

fig.colorbar(surf)
axes[0].title.set_text('Original')
axes[1].title.set_text('Flatten')

plt.show()


plt.plot(r_arr, ws_arr, 'r')
plt.plot(r_arr, gauss(r_arr, *popt), 'b')
plt.show()

fig, ax = plt.subplots(1,1, subplot_kw={"projection": "3d"})

surf = ax.plot_surface(Y_squared, Z_squared, U_squared, cmap=cm.Spectral,
                       linewidth=0, antialiased=False)

fig.colorbar(surf)
ax.title.set_text('Wake deficit in rectangular view')

plt.show()
'''
