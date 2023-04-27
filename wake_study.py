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


def get_U_squared(U2s, Y, Z, fixed_grid=False, toll=1e-1):
    u_min = np.amin(U2s)
    idx_min = [U2s == u_min]
    y_c = Y[idx_min[0]]
    z_c = Z[idx_min[0]]
    z_arr = Z[:, 0]
    y_arr = Y[0, :]

    if fixed_grid:
        # create a 65x50 grid centered in yc, zc
        z_squared = z_arr[0:50]
        idx_yc = np.where(y_arr == y_c)[0][0]
        y_squared = y_arr[idx_yc - 32:idx_yc + 33]
        U_squared = U2s[0:50, idx_yc - 32:idx_yc + 33]

    else:
        idx_rect = [U2s <= -toll]
        y_rect = np.unique(Y[idx_rect[0]])
        z_rect = np.unique(Z[idx_rect[0]])
        ymin = y_rect[0]
        ymax = y_rect[-1]
        zmin = z_rect[0] #start from z=0
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
        jmin = [0]  # always start from z=0
        jmax = np.where(abs(z_arr - (z_c + rz)) < 1e-4)[0]

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

def gauss_2d(xy,A,x0,y0,sigmax,sigmay):
    return A * np.exp(-0.5*(((xy[0]-x0)/(sigmax)) ** 2.0 + ((xy[1]-y0)/(sigmay)) ** 2.0))

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

def gauss_interp_2d(Y_squared, Z_squared, y_c, href, U_squared):
    # in order to center the coordinates in the new reference system centered in yc, href
    U_squared = np.abs(U_squared)
    Z_centered = np.add(Z_squared, -href)
    Y_centered = np.add(Y_squared, -y_c)

    yz_arr = np.vstack((Y_centered.ravel(), Z_centered.ravel()))
    u_arr = U_squared.ravel()

    popt, pcov = curve_fit(gauss_2d, yz_arr, u_arr)

    perr = np.sqrt(np.diag(pcov))  # calculate stdv errors on parameters A, offset, sigma

    return popt, perr, yz_arr, u_arr

def get_field_slices(Y_squared, Z_squared, U_squared, yc, zc):
    dimz, dimy = np.shape(U_squared)
    y_arr = Y_squared[0, :]
    z_arr = Z_squared[:, 0]
    idx_yc = np.where(y_arr == yc)[0]
    idx_zc = np.where(z_arr == zc)[0]
    u_0 = U_squared[:, idx_yc]
    u_90 = U_squared[idx_zc, :].T
    return u_0.flatten(), u_90.flatten()

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------Farm and parameters definition------------------------------------------------------

fi = FlorisInterface('Configurations/gch.yaml')
fi.reinitialize(turbine_library_path='Configurations/turbine_library')
fi.reinitialize(turbine_type=['Baseline_10MW_0'])

#Definition of constants in code
D = fi.floris.farm.rotor_diameters[0]
href = fi.floris.farm.hub_heights[0]
alfa = fi.floris.flow_field.wind_shear
layout_x = [0.]
layout_y = [0.]

solver_settings = {
    "type": "turbine_grid",
    "turbine_grid_points": 3
}

fi.reinitialize(solver_settings=solver_settings, reference_wind_height=href)
fi.reinitialize(layout_x=layout_x, layout_y=layout_y)

#parameters for which the field must be calculated
offset_arr = np.array([-0.5, -0.25, 0, 0.25, 0.5])
DD_arr = np.array([3, 4, 5, 6, 7])
TI_arr = np.array([0.02, 0.06, 0.1])
yaw_arr = np.array([-25, -15, 0, 15, 25])
Vm_arr = np.array([3, 5, 7, 9, 11, 13, 15])
der_arr = np.array([0, 2.5, 5, 10, 15])

l_o = len(offset_arr)
l_d = len(DD_arr)
l_t = len(TI_arr)
l_y = len(yaw_arr)
l_v = len(Vm_arr)


m_d = l_v * l_t * l_y * l_d * l_o
m_v = l_t * l_y * l_d * l_o
m_t = l_y * l_d * l_o
m_y = l_d * l_o
m_dd = l_o


#--------------------------------------------circular wake--------------------------------------------------------------
#definition of storing numpy matrix
gauss_parameters = np.zeros((np.shape(der_arr)[0] * np.shape(Vm_arr)[0] * np.shape(TI_arr)[0] * np.shape(yaw_arr)[0] * np.shape(offset_arr)[0] * np.shape(DD_arr)[0], 19))

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------Loop begins over parameters---------------------------------------------------------


start_time = time.time()
for i_d, der in enumerate(der_arr):
    if der == 2.5:
        turbine_name = 'Baseline_10MW_2.5.yaml'
        fi.reinitialize(turbine_type=[turbine_name])
    else:
        der = str(int(der))
        turbine_name = 'Baseline_10MW_' + der + '.yaml'
        fi.reinitialize(turbine_type=[turbine_name])
        der = float(der)

    for i_v, Vm in enumerate(Vm_arr):
        fi.reinitialize(wind_speeds=[Vm])

        for i_t, TI in enumerate(TI_arr):
            fi.reinitialize(turbulence_intensity=TI)

            for i_y, yaw_angle in enumerate(yaw_arr):
                yaw_angles = np.zeros((1, 1, 1))
                yaw_angles[0, 0, 0] = yaw_angle
                fi.calculate_wake(yaw_angles=yaw_angles)

                for i_dd, dd in enumerate(DD_arr):
                    cross_plane = fi.calculate_cross_plane(
                        y_resolution=101,
                        z_resolution=101,
                        downstream_dist=dd * D,
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

                    U_new = flatten_wind_field(Z, U, alfa, Vm, href)
                    U_squared, Y_squared, Z_squared, y_c, z_c = get_U_squared(U_new, Y, Z)
                    popt, perr, R_flat, U_flat = gauss_interp(Y_squared, Z_squared, y_c, z_c, U_squared)
                    z_shape = np.shape(U_squared)[0]
                    y_shape = np.shape(U_squared)[1]

                    A, mu, sigma = popt

                    A = round(A, 3)
                    sigma = round(sigma, 3)

                    interp_values = np.array(gauss(R_flat, *popt))
                    dev = np.abs(np.add(interp_values, -abs(U_flat)))
                    err_max = round(np.amax(dev), 3)
                    err_mean = round(np.mean(dev), 3)
                    stdv = round(np.std(dev), 3)
                    dev_rel = np.divide(dev, Vm/100)
                    err_rel_max = round(np.amax(dev_rel), 3)
                    err_rel_mean = round(np.mean(dev_rel), 3)
                    stdv_rel = round(np.std(dev_rel), 3)


                    for i_o, offset in enumerate(offset_arr):
                        y_c_d = round(y_c[0] - offset * D, 3)
                        popti = np.array([der, Vm, TI, yaw_angle, dd, offset, A, y_c_d, z_c[0], sigma, Vm, err_max, err_mean,
                                          stdv, err_rel_max, err_rel_mean, stdv_rel, y_shape, z_shape])
                        gauss_parameters[i_d * m_d + i_v * m_v + i_t * m_t + i_y * m_y + i_dd * m_dd + i_o, :] = popti


everything = pd.DataFrame(gauss_parameters, columns=['der', 'Vm', 'TI', 'yaw', 'DD', 'offset', 'peak', 'yc_d', 'zc_d', 'sigma', 'Vm', 'e_max', 'e_mean', 'e_stdv', 'e_rel_max', 'e_rel_mean', 'e_rel_stdv', 'y_shape', 'z_shape'])

filepath = Path('C:/Users/randr/Desktop/wake_circular ')

'''

#--------------------------------------------elliptical wake--------------------------------------------------------------
#definition of storing numpy matrix
gauss_parameters = np.zeros((np.shape(der_arr)[0] * np.shape(Vm_arr)[0] * np.shape(TI_arr)[0] * np.shape(yaw_arr)[0] * np.shape(offset_arr)[0] * np.shape(DD_arr)[0], 20))

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------Loop begins over parameters---------------------------------------------------------


start_time = time.time()
for i_d, der in enumerate(der_arr):
    if der == 2.5:
        turbine_name = 'Baseline_10MW_2.5.yaml'
        fi.reinitialize(turbine_type=[turbine_name])
    else:
        der = str(int(der))
        turbine_name = 'Baseline_10MW_' + der + '.yaml'
        fi.reinitialize(turbine_type=[turbine_name])
        der = float(der)

    for i_v, Vm in enumerate(Vm_arr):
        fi.reinitialize(wind_speeds=[Vm])

        for i_t, TI in enumerate(TI_arr):
            fi.reinitialize(turbulence_intensity=TI)

            for i_y, yaw_angle in enumerate(yaw_arr):
                yaw_angles = np.zeros((1, 1, 1))
                yaw_angles[0, 0, 0] = yaw_angle
                fi.calculate_wake(yaw_angles=yaw_angles)

                for i_dd, dd in enumerate(DD_arr):
                    cross_plane = fi.calculate_cross_plane(
                        y_resolution=101,
                        z_resolution=101,
                        downstream_dist=dd * D,
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

                    U_new = flatten_wind_field(Z, U, alfa, Vm, href)
                    U_squared, Y_squared, Z_squared, y_c, z_c = get_U_squared_no_z0(U_new, Y, Z)
                    popt, perr, yz_arr, u_arr = gauss_interp_2d(Y_squared, Z_squared, y_c, z_c, U_squared)
                    z_shape = np.shape(U_squared)[0]
                    y_shape = np.shape(U_squared)[1]

                    A, y_gc, z_gc, sigmay, sigmaz = popt

                    A = round(A, 3)
                    y_gc = round(y_gc, 3)
                    z_gc = round(z_gc, 3)
                    sigmay = round(sigmay, 3)
                    sigmaz = round(sigmaz, 3)

                    interp_values = np.array(gauss_2d(yz_arr, *popt))
                    dev = np.abs(np.add(interp_values, -abs(u_arr)))
                    err_max = round(np.amax(dev), 3)
                    err_mean = round(np.mean(dev), 3)
                    stdv = round(np.std(dev), 3)
                    dev_rel = np.divide(dev, Vm/100)
                    err_rel_max = round(np.amax(dev_rel), 3)
                    err_rel_mean = round(np.mean(dev_rel), 3)
                    stdv_rel = round(np.std(dev_rel), 3)


                    for i_o, offset in enumerate(offset_arr):
                        y_c_d = round(y_c[0] + y_gc - offset * D, 3)
                        z_c_d = round(z_c[0] + z_gc, 3)
                        popti = np.array([der, Vm, TI, yaw_angle, dd, offset, A, y_c_d, z_c_d, sigmay, sigmaz, Vm, err_max, err_mean,
                                          stdv, err_rel_max, err_rel_mean, stdv_rel, y_shape, z_shape])
                        gauss_parameters[i_d * m_d + i_v * m_v + i_t * m_t + i_y * m_y + i_dd * m_dd + i_o, :] = popti


everything = pd.DataFrame(gauss_parameters, columns=['der', 'Vm', 'TI', 'yaw', 'DD', 'offset', 'peak', 'yc_d', 'zc_d', 'sigmay', 'sigmaz', 'Vm', 'e_max', 'e_mean', 'e_stdv', 'e_rel_max', 'e_rel_mean', 'e_rel_stdv', 'y_shape', 'z_shape'])

filepath = Path('C:/Users/randr/Desktop/wake_elliptical_noz0')
'''
filepath.parent.mkdir(parents=True, exist_ok=True)
everything.to_csv(filepath, index=False)


end_time = time.time()
time_elapsed = end_time - start_time

print(f'Process took {time_elapsed} seconds')


print('-----------------------------------------------------------------------------------------------------------------')




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
