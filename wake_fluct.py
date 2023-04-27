import matplotlib.pyplot as plt
from matplotlib import cm
import time
from pathlib import Path
import os
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

    return popt, perr, R_flat, U_flat

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

Vm = 11.4
DD = 7
fi.reinitialize(solver_settings=solver_settings, reference_wind_height=href)
fi.reinitialize(layout_x=layout_x, layout_y=layout_y)
fi.reinitialize(wind_speeds=[Vm])
fi.reinitialize(turbulence_intensity=0.1)
Vref = fi.floris.flow_field.wind_speeds[0]


yaw_angles = np.zeros((1,1,1))
yaw_angles[0,0,0] = 0


TI_arr = [0.02, 0.06, 0.1]
DD_arr = [3, 4, 5, 6, 7]

for TI in TI_arr:
    fi.reinitialize(turbulence_intensity=TI)
    for DD in DD_arr:
        cross_plane = fi.calculate_cross_plane(
            y_resolution=101,
            z_resolution=101,
            downstream_dist=DD * D,
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

        U2s = flatten_wind_field(Z, U, alfa, Vref, href)

        U_squared, Y_squared, Z_squared, y_c, z_c = get_U_squared_no_z0(U2s, Y, Z)
        popt, perr, r_arr, ws_arr = gauss_interp(Y_squared, Z_squared, y_c, z_c, U_squared)

        interp_values = np.array(gauss(r_arr, *popt))
        dev = np.abs(np.add(interp_values, -ws_arr))
        err_max = round(np.amax(dev), 3)

        plt.plot(r_arr, ws_arr, 'r', label='data')
        plt.plot(r_arr, interp_values, 'b', label='interpolated')
        plt.title('TI = {}, DD = {}, \nerr_max = {} m/s'.format(TI, DD, err_max))
        plt.xlabel('distance from wake center [m]')
        plt.ylabel('wake deficit [m/s]')
        plt.legend()
        direct_path = 'C:/Users/randr/Desktop/Polimi/5° Anno/2° semestre/Tesi/Reports/TI_DD_noz0'
        figname = (str(TI) + '_' + str(DD) + '.png')
        filepath = os.path.join(direct_path, figname)

        plt.savefig(filepath)
        plt.clf()



'''
cross_plane = fi.calculate_cross_plane(
    y_resolution=101,
    z_resolution=101,
    downstream_dist=DD * D,
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

U2s = flatten_wind_field(Z, U, alfa, Vref, href)
U_squared, Y_squared, Z_squared, y_c, z_c = get_U_squared_no_z0(U2s, Y, Z)


#circular gaussian
popt_c, _, r_arr, ws_arr = gauss_interp(Y_squared, Z_squared, y_c, z_c, U_squared)
interp_c = np.array(gauss(r_arr, *popt_c))
dev = np.abs(np.add(interp_c, -ws_arr))
err_max_c = round(np.amax(dev), 3)
err_mean_c = round(np.mean(dev), 3)


#ellittical gaussian
popt_e, _, yz_arr, u_arr = gauss_interp_2d(Y_squared, Z_squared, y_c, z_c, U_squared)
interp_e = np.array(gauss_2d(yz_arr, *popt_e))
dev = np.abs(np.add(interp_e, -u_arr))
err_max_e = round(np.amax(dev), 3)
err_mean_e = round(np.mean(dev), 3)

interp_matrix = np.reshape(interp_e, np.shape(Y_squared))

u_0, u_90 = get_field_slices(Y_squared, Z_squared, U_squared, y_c, z_c)
interp_e_0, interp_e_90 = get_field_slices(Y_squared, Z_squared, interp_matrix, y_c, z_c)

z_arr = Z_squared[:, 0]
y_arr = Y_squared[0, :]

z_u = z_c + D/2
z_u = z_arr[(np.abs(z_arr - z_u)).argmin()]
z_d = z_c - D/2
z_d = z_arr[(np.abs(z_arr - z_d)).argmin()]

y_sx = y_c - D/2
y_sx = y_arr[(np.abs(y_arr - y_sx)).argmin()]
y_dx = y_c + D/2
y_dx = y_arr[(np.abs(y_arr - y_dx)).argmin()]

u_sx, u_u = get_field_slices(Y_squared, Z_squared, U_squared, y_sx, z_u)
u_dx, u_d = get_field_slices(Y_squared, Z_squared, U_squared, y_dx, z_d)

interp_e_sx, interp_e_u = get_field_slices(Y_squared, Z_squared, interp_matrix, y_sx, z_u)
interp_e_dx, interp_e_d = get_field_slices(Y_squared, Z_squared, interp_matrix, y_dx, z_d)

r_0 = np.sqrt(np.add(np.square(np.add(z_arr, -z_c)), np.square(0)))
r_sx = np.sqrt(np.add(np.square(np.add(z_arr, -z_c)), np.square(np.add(y_sx, -y_c))))
r_dx = np.sqrt(np.add(np.square(np.add(z_arr, -z_c)), np.square(np.add(y_dx, -y_c))))
r_90 = np.sqrt(np.add(np.square(np.add(y_arr, -y_c)), np.square(0)))
r_d = np.sqrt(np.add(np.square(np.add(y_arr, -y_c)), np.square(np.add(z_d, -z_c))))
r_u = np.sqrt(np.add(np.square(np.add(y_arr, -y_c)), np.square(np.add(z_u, -z_c))))

interp_c_0 = np.array(gauss(r_0, *popt_c))
interp_c_90 = np.array(gauss(r_90, *popt_c))
interp_c_sx = np.array(gauss(r_sx, *popt_c))
interp_c_dx = np.array(gauss(r_dx, *popt_c))
interp_c_u = np.array(gauss(r_u, *popt_c))
interp_c_d = np.array(gauss(r_d, *popt_c))


plt.plot(y_arr, abs(u_90), 'r', label='data')
plt.plot(y_arr, interp_e_90, 'b', label='interpolated elliptical values')
plt.title('Horizontal cut through center')
plt.xlabel('y [m]')
plt.ylabel('wake deficit [m/s]')
plt.legend()
plt.show()

plt.plot(y_arr, abs(u_u), 'r', label='data')
plt.plot(y_arr, interp_e_u, 'b', label='interpolated elliptical values')
plt.title('Horizontal cut at z = {} m'.format(round(z_u,2)))
plt.xlabel('y [m]')
plt.ylabel('wake deficit [m/s]')
plt.legend()
plt.show()

plt.plot(y_arr, abs(u_d), 'r', label='data')
plt.plot(y_arr, interp_e_d, 'b', label='interpolated elliptical values')
plt.title('Horizontal cut at z = {} m'.format(round(z_d,2)))
plt.xlabel('y [m]')
plt.ylabel('wake deficit [m/s]')
plt.legend()
plt.show()

plt.plot(z_arr, abs(u_0), 'r', label='data')
plt.plot(z_arr, interp_e_0, 'b', label='interpolated elliptical values')
plt.title('Vertical cut through center')
plt.xlabel('z [m]')
plt.ylabel('wake deficit [m/s]')
plt.legend()
plt.show()

plt.plot(z_arr, abs(u_sx), 'r', label='data')
plt.plot(z_arr, interp_e_sx, 'b', label='interpolated elliptical values')
plt.title('Vertical cut at y = {} m'.format(round(y_sx,2)))
plt.xlabel('z [m]')
plt.ylabel('wake deficit [m/s]')
plt.legend()
plt.show()

plt.plot(z_arr, abs(u_dx), 'r', label='data')
plt.plot(z_arr, interp_e_dx, 'b', label='interpolated elliptical values')
plt.title('Vertical cut at y = {} m'.format(round(y_dx,2)))
plt.xlabel('z [m]')
plt.ylabel('wake deficit [m/s]')
plt.legend()
plt.show()

#-----------------------circular-----------------------------------------------------
plt.plot(y_arr, abs(u_90), 'r', label='data')
plt.plot(y_arr, interp_c_90, 'b', label='interpolated circular values')
plt.title('Horizontal cut through center')
plt.xlabel('y [m]')
plt.ylabel('wake deficit [m/s]')
plt.legend()
plt.show()

plt.plot(y_arr, abs(u_u), 'r', label='data')
plt.plot(y_arr, interp_c_u, 'b', label='interpolated circular values')
plt.title('Horizontal cut at z = {} m'.format(round(z_u,2)))
plt.xlabel('y [m]')
plt.ylabel('wake deficit [m/s]')
plt.legend()
plt.show()

plt.plot(y_arr, abs(u_d), 'r', label='data')
plt.plot(y_arr, interp_c_d, 'b', label='interpolated circular values')
plt.title('Horizontal cut at z = {} m'.format(round(z_d,2)))
plt.xlabel('y [m]')
plt.ylabel('wake deficit [m/s]')
plt.legend()
plt.show()

plt.plot(z_arr, abs(u_0), 'r', label='data')
plt.plot(z_arr, interp_c_0, 'b', label='interpolated circular values')
plt.title('Vertical cut through center')
plt.xlabel('z [m]')
plt.ylabel('wake deficit [m/s]')
plt.legend()
plt.show()

plt.plot(z_arr, abs(u_sx), 'r', label='data')
plt.plot(z_arr, interp_c_sx, 'b', label='interpolated circular values')
plt.title('Vertical cut at y = {} m'.format(round(y_sx,2)))
plt.xlabel('z [m]')
plt.ylabel('wake deficit [m/s]')
plt.legend()
plt.show()

plt.plot(z_arr, abs(u_dx), 'r', label='data')
plt.plot(z_arr, interp_c_dx, 'b', label='interpolated circular values')
plt.title('Vertical cut at y = {} m'.format(round(y_dx,2)))
plt.xlabel('z [m]')
plt.ylabel('wake deficit [m/s]')
plt.legend()
plt.show()

err_max_c_0 = np.amax(np.abs(np.add(interp_c_0, u_0)))
err_max_c_90 = np.amax(np.abs(np.add(interp_c_90, u_90)))
err_max_c_sx = np.amax(np.abs(np.add(interp_c_sx, u_sx)))
err_max_c_dx = np.amax(np.abs(np.add(interp_c_dx, u_dx)))
err_max_c_u = np.amax(np.abs(np.add(interp_c_u, u_u)))
err_max_c_d = np.amax(np.abs(np.add(interp_c_d, u_d)))
err_max_c_arr = [err_max_c_0, err_max_c_90, err_max_c_sx, err_max_c_dx, err_max_c_u, err_max_c_d]
err_max_c = np.amax(err_max_c_arr)

err_max_e_0 = np.amax(np.abs(np.add(interp_e_0, u_0)))
err_max_e_90 = np.amax(np.abs(np.add(interp_e_90, u_90)))
err_max_e_sx = np.amax(np.abs(np.add(interp_e_sx, u_sx)))
err_max_e_dx = np.amax(np.abs(np.add(interp_e_dx, u_dx)))
err_max_e_u = np.amax(np.abs(np.add(interp_e_u, u_u)))
err_max_e_d = np.amax(np.abs(np.add(interp_e_d, u_d)))
err_max_e_arr = [err_max_e_0, err_max_e_90, err_max_e_sx, err_max_e_dx, err_max_e_u, err_max_e_d]
err_max_e = np.amax(err_max_e_arr)

print(err_max_c_arr)
print(err_max_c)
print('')
print(err_max_e_arr)
print(err_max_e)
'''





'''
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1,2,1, projection = '3d')
surf = ax.plot_surface(Y_centered, Z_centered, U_squared, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf)

ax.title.set_text('Data')

ax = fig.add_subplot(1,2,2, projection = '3d')
ax.title.set_text('Interpolated')

surf2 = ax.plot_surface(Y_centered, Z_centered, interp_matrix, cmap=cm.coolwarm,
                             linewidth=0, antialiased = False)

fig.colorbar(surf2)

plt.show()
'''

'''
fig, axes = plt.subplots(1, 2)
axes[0].plot(z_arr, u_0)

axes[1].plot(y_arr, u_90)
plt.show()
'''


'''

TI_arr = [0.02, 0.06, 0.1]
DD_arr = [3, 4, 5, 6, 7]

for TI in TI_arr:
    fi.reinitialize(turbulence_intensity=TI)
    for DD in DD_arr:
        cross_plane = fi.calculate_cross_plane(
            y_resolution=101,
            z_resolution=101,
            downstream_dist=DD * D,
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

        U2s = flatten_wind_field(Z, U, alfa, Vref, href)

        U_squared, Y_squared, Z_squared, y_c, z_c = get_U_squared(U2s, Y, Z)
        popt, perr, r_arr, ws_arr = gauss_interp(Y_squared, Z_squared, y_c, z_c, U_squared)

        interp_values = np.array(gauss(r_arr, *popt))
        dev = np.abs(np.add(interp_values, -ws_arr))
        err_max = round(np.amax(dev), 3)

        plt.plot(r_arr, ws_arr, 'r')
        plt.plot(r_arr, interp_values, 'b')
        plt.title('err_max=' + str(err_max) + ' m/s')
        plt.xlabel('distance from wake center [m]')
        plt.ylabel('wake deficit [m/s]')
        direct_path = 'C:/Users/randr/Desktop/Polimi/5° Anno/2° semestre/Tesi/Reports/TI_DD'
        figname = (str(TI) + '_' + str(DD) + '.png')
        filepath = os.path.join(direct_path, figname)

        plt.savefig(filepath)
        plt.clf()



fig, axes = plt.subplots(1,1, subplot_kw={"projection": "3d"})

surf = axes.plot_surface(Y, Z, U, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf)

axes.view_init(roll = 90)

axes.title.set_text('Original')

plt.show()

fig, axes = plt.subplots(1,1, subplot_kw={"projection": "3d"})

axes.title.set_text('Flattened')

surf2 = axes.plot_surface(Y, Z, U_new, cmap=cm.coolwarm,
                             linewidth=0, antialiased = False)
axes.view_init(roll = 90)

fig.colorbar(surf2)


plt.show()



plt.plot(r_arr, ws_arr, 'r')
plt.plot(r_arr, gauss(r_arr, *popt), 'b')
plt.show()

interp = np.array(gauss(r_arr, *popt))
dev = np.abs(np.add(interp, -ws_arr))
err_mean = np.mean(dev)
err_max = np.amax(abs(dev))
stdv = np.std(dev)
sigma_u = np.sqrt(np.divide(np.sum(np.square(dev)), len(dev)))


print(err_mean)
print(err_max)
print(stdv)
print(sigma_u)



plt.plot(r_arr, dev)
plt.show()

'''