from pathlib import Path
import numpy as np
import pandas as pd
import math
from scipy.optimize import curve_fit
from floris.tools import FlorisInterface


#some header to test git
#this is one of the main files
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------Functions for gauss parameters calculation------------------------------------------
def flatten_wind_field(Z, U, alfa, Vref, href):
    '''
    Subtracts the mean wind speed at each height calculated through an exponential shear function, in order to highlight
    the wake deficit for each grid point.
    :param Z: vertical coordinates of the grid points [m]
    :param U: mean wind speed of the grid points along the main wind direction [m/s]
    :param alfa: shear coefficient [-]
    :param Vref: mean wind speed at hub height [m/s]
    :param href: hub height [m]
    :return: matrix of same dimensions as U to which the mean wind speed is subtracted
    '''
    z_arr = Z[:,0]
    U_new = np.zeros(np.shape(U))
    for i, z in enumerate(z_arr):
        ws = np.ones(np.shape(U)[1])
        shear = Vref * pow((z/href), alfa)
        ws = np.multiply(ws, shear)
        U_new[i,:] = U[i,:] - ws

    return U_new

def get_U_squared(U2s, Y, Z, fixed_grid=False, toll=1e-1):
    '''
    Selection of the relevant grid points. The original wake field is restricted to consider only points that have a steady
    state wake deficit higher than toll [m/s]
    :param U2s: matrix of wake deficit to reduce (square) [m/s]
    :param Y: lateral coordinates of the grid points [m]
    :param Z: vertical coordinates of the grid points [m]
    :param fixed_grid: if True the output matrix has fixed dimensions (65x51) and it is centered in the wake center
    :param toll: threshold under which the point is excluded from the wake region
    :return:
            U_squared: wake deficit for the selected points [m/s]
            Y_squared: lateral coordinates of the selected points [m]
            Z_squared: vertical coordinates of the selected points [m]
            y_c: lateral coordinate of the wake center [m]
            z_c: vertical coordinate of the wake center [m]
            flag_no_deficit: True if no wake deficit is found (too small)
    '''
    u_min = np.amin(U2s)
    idx_min = [U2s == u_min]
    y_c = Y[idx_min[0]]
    z_c = Z[idx_min[0]]
    z_arr = Z[:, 0]
    y_arr = Y[0, :]
    flag_no_deficit = 0

    if fixed_grid:
        #create a 65x51 grid centered in yc, zc
        idx_yc = np.where(y_arr == y_c)[0][0]
        idx_zc = np.where(z_arr == z_c)[0][0]
        y_squared = y_arr[idx_yc-32:idx_yc+33]
        z_squared = z_arr[np.amax(0, idx_zc-25):idx_zc+26]
        U_squared = U2s[np.amax(0, idx_zc-25):idx_zc+26, idx_yc-32:idx_yc+33]

    else:
        idx_rect = [U2s <= -toll]
        y_rect = np.unique(Y[idx_rect[0]])
        z_rect = np.unique(Z[idx_rect[0]])
        if len(y_rect) == 0 or len(z_rect) == 0:
            flag_no_deficit = 1
            U_squared = []
            Y_squared = []
            Z_squared = []
            return U_squared, Y_squared, Z_squared, y_c, z_c, flag_no_deficit
        else:
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

    return U_squared, Y_squared, Z_squared, y_c, z_c, flag_no_deficit


def gauss_2d(yz,A,y0,z0,sigmay,sigmaz):
    '''
    Elliptical based gaussian function
    :param yz: 2xN array containing the coordinates of the points in which the function is computed [m]
    :param A: gaussian max value [m/s]
    :param y0: lateral coordinate of the wake center [m]
    :param z0: vertical coordinate of the wake center [m]
    :param sigmay: standard deviation of the gaussian along the lateral direction [m]
    :param sigmaz: standard deviation of the gaussian along the vertical direction [m]
    :return: value of the function in m/s
    '''
    return A * np.exp(-0.5*(((yz[0]-y0)/(sigmay)) ** 2.0 + ((yz[1]-z0)/(sigmaz)) ** 2.0))

def gauss_c(yz, A, y0, z0, sigma):
    '''
    Circular based gaussian function
    :param yz: 2xN array containing the coordinates of the points in which the function is computed [m]
    :param A: gaussian max value [m/s]
    :param y0: lateral coordinate of the wake center [m]
    :param z0: vertical coordinate of the wake center [m]
    :param sigma: standard deviation of the gaussian [m]
    :return: value of the function in m/s
    '''
    return A * np.exp(-0.5*(((yz[0]-y0)/(sigma)) ** 2.0 + ((yz[1]-z0)/(sigma)) ** 2.0))


def gauss_interp_2d(Y_squared, Z_squared, y_c, href, U_squared):
    '''
    Fits the wake deficit into a elliptical based gaussian function
    :param Y_squared: lateral coordinates of the selected points [m]
    :param Z_squared: vertical coordinates of the selected points [m]
    :param y_c: lateral coordinate of the wake center [m]
    :param href: hub height [m]
    :param U_squared: wake deficit of the selected points [m/s]
    :return:
            popt: containing the parameters of the gaussian function
            perr: containing the standard deviations for the fitted parameters
            yz_arr: 2xN array containing the coordinates of the selected points [m]
            u_arr: array containing the wake deficit of the selected points [m/s]
    '''
    # in order to center the coordinates in the new reference system centered in yc, href
    U_squared = np.abs(U_squared)
    Z_centered = np.add(Z_squared, -href)
    Y_centered = np.add(Y_squared, -y_c)

    yz_arr = np.vstack((Y_centered.ravel(), Z_centered.ravel()))
    u_arr = U_squared.ravel()

    popt, pcov = curve_fit(gauss_2d, yz_arr, u_arr)

    perr = np.sqrt(np.diag(pcov))  # calculate stdv errors on parameters A, offset, sigma

    return popt, perr, yz_arr, u_arr

def gauss_interp_c(Y_squared, Z_squared, y_c, href, U_squared):
    '''
    Fits the wake deficit into a circular based gaussian function
    :param Y_squared: lateral coordinates of the selected points [m]
    :param Z_squared: vertical coordinates of the selected points [m]
    :param y_c: lateral coordinate of the wake center [m]
    :param href: hub height [m]
    :param U_squared: wake deficit of the selected points [m/s]
    :return:
            popt: containing the parameters of the gaussian function
            perr: containing the standard deviations for the fitted parameters
            yz_arr: 2xN array containing the coordinates of the selected points [m]
            u_arr: array containing the wake deficit of the selected points [m/s]
    '''
    # in order to center the coordinates in the new reference system centered in yc, href
    U_squared = np.abs(U_squared)
    Z_centered = np.add(Z_squared, -href)
    Y_centered = np.add(Y_squared, -y_c)

    yz_arr = np.vstack((Y_centered.ravel(), Z_centered.ravel()))
    u_arr = U_squared.ravel()

    popt, pcov = curve_fit(gauss_c, yz_arr, u_arr)

    perr = np.sqrt(np.diag(pcov))  # calculate stdv errors on parameters A, offset, sigma

    return popt, perr, yz_arr, u_arr

def get_field_slices(Y_squared, Z_squared, U_squared, yc, zc):
    '''
    Returns the wake deficit along horizontal and vertical slices passing through the wake center
    :param Y_squared: lateral coordinates of the selected points [m]
    :param Z_squared: vertical coordinates of the selected points [m]
    :param U_squared: wake deficit of the selected points [m/s]
    :param yc: lateral coordinate of the wake center [m]
    :param zc: vertical coordinate of the wake center [m]
    :return:
            u_0: array containing wake deficit along the vertical slice [m/s]
            u_90: array containing wake deficit along the horizontal slice [m/s]
    '''
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

fi = FlorisInterface('Configurations/gch.yaml') #Gaussian Curl Hybrid model
fi.reinitialize(turbine_library_path='Configurations/turbine_library')
fi.reinitialize(turbine_type=['Baseline_10MW_0'])

#Definition of constants in code
D = fi.floris.farm.rotor_diameters[0]
href = fi.floris.farm.hub_heights[0]
alfa = fi.floris.flow_field.wind_shear
layout_x = [0.]
layout_y = [0.]

#layout definition
fi.reinitialize(reference_wind_height=href, layout_x=layout_x, layout_y=layout_y)

#parameters for which the field must be calculated: lateral offset, downstream distance,
#yaw angle, wind speed and derating value. Distances are non-dimensionalized with respect to rotor diameter
offset_arr = np.array([-0.5, -0.25, 0, 0.25, 0.5])
DD_arr = np.array([5])
yaw_arr = np.array([-25, -15, 0, 15, 25])
Vm_arr = np.array([4, 5, 7, 9, 11, 13, 15])
der_arr = np.array([0, 2.5, 5, 10, 15])


l_o = len(offset_arr)
l_dd = len(DD_arr)
l_y = len(yaw_arr)
l_v = len(Vm_arr)
l_der = len(der_arr)

m_der = l_v * l_y * l_dd * l_o
m_v = l_y * l_dd * l_o
m_y = l_dd * l_o
m_dd = l_o


#--------------------------------------------circular wake--------------------------------------------------------------
#definition of storing numpy matrix
gauss_parameters = np.zeros((l_der * l_v * l_y * l_o * l_dd, 16))

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------Loop begins over parameters---------------------------------------------------------


for i_der, der in enumerate(der_arr):
    if der % 1 > 0:
        der = str(der)
    else:
        der = str(int(der))
    turbine_name = 'Baseline_10MW_' + der + '.yaml'
    fi.reinitialize(turbine_type=[turbine_name])
    der = float(der)

    for i_v, Vm in enumerate(Vm_arr):
        fi.reinitialize(wind_speeds=[Vm])
        TI_amb = round(0.16 * (Vm * 0.75 + 5.6)/Vm, 3) #TI for class 1A IEC61400
        fi.reinitialize(turbulence_intensity=TI_amb)

        for i_y, yaw_angle in enumerate(yaw_arr):
            yaw_angles = np.zeros((1, 1, 1))
            yaw_angles[0, 0, 0] = yaw_angle
            fi.calculate_wake(yaw_angles=yaw_angles)
            CT = fi.get_turbine_Cts()[0,0,0]
            CT = round(CT, 3)
            for i_dd, dd in enumerate(DD_arr):
                #calculate wind field at the specified downstream distance
                cross_plane = fi.calculate_cross_plane(
                    y_resolution=101,
                    z_resolution=101,
                    downstream_dist=dd * D,
                    yaw_angles=yaw_angles
                )

                #calculation of additional turbulence intensity and wake turbulence intensity with the
                #Quarton-Ainslie model
                m = 1 / np.sqrt(1 - CT)
                r0 = D/2 * np.sqrt((1 + m) / 2)
                drdx_amb = 2.5 * TI_amb + 0.005
                drdx_sh = (1 - m) * np.sqrt(1.49 + m) / (9.76 * (1 + m))
                drdx_mech = 0.012 * 3 * 8
                drdx = np.sqrt(drdx_amb ** 2 + drdx_sh ** 2 + drdx_mech ** 2)
                Xn = (np.sqrt(0.214 + 0.144 * m) * (1 - np.sqrt(0.134 + 0.124 * m))) / \
                     ((1 - np.sqrt(0.214 + 0.144 * m)) * (np.sqrt(0.134 + 0.124 * m))) * r0 / drdx

                TI_QA = round((4.8 * CT ** (0.7) * (TI_amb * 100) ** (0.68) * (dd * D / Xn) ** (-0.57))/100, 3)
                TI_tot = round(np.sqrt(np.sum([np.square(TI_amb), np.square(TI_QA)])), 3)

                #post processing of the wind field to obtain gaussian parameters
                df = cross_plane.df
                y_grid = np.array(df['x1'])
                z_grid = np.array(df['x2'])
                u_grid = np.array(df['u'])

                n = int(math.sqrt(u_grid.shape[0]))
                Y = y_grid.reshape(n, n)
                Z = z_grid.reshape(n, n)
                U = u_grid.reshape(n, n)
                U_new = flatten_wind_field(Z, U, alfa, Vm, href)
                U_squared, Y_squared, Z_squared, y_c, z_c, flag_no_deficit = get_U_squared(U_new, Y, Z)

                #in order to account for field with negligible wake deficits, otherwise error in the fitting function
                if flag_no_deficit:
                    A = 0
                    sigma = 0
                    y_gc = 0
                    z_gc = 0
                    err_max = np.amin(abs(U_new))
                    err_mean = 0
                else:
                    u_max = abs(np.amin(U_squared))

                    if u_max<0.38:
                        #if maximum wake deficit is too small, the fitting function does not work well so it is scaled up,
                        #fitted and then the peak of the gaussian is scaled back down again
                        U_squared = np.multiply(U_squared, 0.38/u_max)
                        popt, perr, yz_flat, U_flat = gauss_interp_c(Y_squared, Z_squared, y_c, z_c, U_squared)
                        popt[0] = popt[0] * u_max/0.38
                        U_flat = np.divide(U_flat, 0.38/u_max)
                    else:
                        popt, perr, yz_flat, U_flat = gauss_interp_c(Y_squared, Z_squared, y_c, z_c, U_squared)

                    A, y_gc, z_gc, sigma = popt

                    A = round(A, 3) #peak of the gaussian function
                    sigma = round(sigma, 3) #standard deviation of the gaussian function
                    y_gc = round(y_gc, 3) #lateral position of the gaussian center
                    z_gc = round(z_gc, 3) #vertical position of the gaussian center

                    #deviation of the gaussian function from the sample points
                    interp_values = np.array(gauss_c(yz_flat, *popt))
                    dev = np.abs(np.add(interp_values, -abs(U_flat)))
                    err_max = round(np.amax(dev), 3)
                    err_mean = round(np.mean(dev), 3)

                for i_o, offset in enumerate(offset_arr):
                    #coordinates of the gaussian center in the reference system centered in the downstream turbine
                    y_c_d = round(y_c[0] + y_gc - offset * D, 2)
                    z_c_d = round(z_c[0] + z_gc, 2)
                    popti = np.array([der, Vm, TI_amb, yaw_angle, dd, offset, A, y_c_d, z_c_d, sigma, Vm, CT, TI_QA,
                                      TI_tot, err_max, err_mean])
                    gauss_parameters[i_der * m_der + i_v * m_v + i_y * m_y + i_dd * m_dd + i_o, :] = popti


wake_properties = pd.DataFrame(gauss_parameters, columns=['der', 'Vm', 'TI', 'yaw', 'DD', 'offset', 'peak', 'yc_d', 'zc_d',
                                                     'sigma', 'Vm', 'CT', 'TI_QA', 'TI_tot', 'e_max', 'e_mean'])

#save wake properties as a csv to this path
filepath = Path('wake_prova.csv')
#filepath = Path('C:/Users/randr/Desktop/wake_5D.csv')


'''
#--------------------------------------------elliptical wake--------------------------------------------------------------
#definition of storing numpy matrix
gauss_parameters = np.zeros((l_der * l_v * l_y * l_o * l_dd, 17))

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------Loop begins over parameters---------------------------------------------------------


start_time = time.time()
for i_der, der in enumerate(der_arr):
    if der % 1 > 0:
        der = str(der)
    else:
        der = str(int(der))
    turbine_name = 'Baseline_10MW_' + der + '.yaml'
    fi.reinitialize(turbine_type=[turbine_name])

    for i_v, Vm in enumerate(Vm_arr):
        fi.reinitialize(wind_speeds=[Vm])
        TI_amb = round(0.16 * (Vm * 0.75 + 5.6) / Vm, 3)  # TI for class 1A IEC61400
        fi.reinitialize(turbulence_intensity=TI_amb)

        for i_y, yaw_angle in enumerate(yaw_arr):
            yaw_angles = np.zeros((1, 1, 1))
            yaw_angles[0, 0, 0] = yaw_angle
            fi.calculate_wake(yaw_angles=yaw_angles)
            CT = fi.get_turbine_Cts()[0, 0, 0]
            CT = round(CT, 3)

            for i_dd, dd in enumerate(DD_arr):
                # calculate wind field at the specified downstream distance
                cross_plane = fi.calculate_cross_plane(
                    y_resolution=101,
                    z_resolution=101,
                    downstream_dist=dd * D,
                    yaw_angles=yaw_angles
                )

                # calculation of additional turbulence intensity and wake turbulence intensity with the
                # Quarton-Ainslie model
                m = 1 / np.sqrt(1 - CT)
                r0 = D / 2 * np.sqrt((1 + m) / 2)
                drdx_amb = 2.5 * TI_amb + 0.005
                drdx_sh = (1 - m) * np.sqrt(1.49 + m) / (9.76 * (1 + m))
                drdx_mech = 0.012 * 3 * 8
                drdx = np.sqrt(drdx_amb ** 2 + drdx_sh ** 2 + drdx_mech ** 2)
                Xn = (np.sqrt(0.214 + 0.144 * m) * (1 - np.sqrt(0.134 + 0.124 * m))) / \
                     ((1 - np.sqrt(0.214 + 0.144 * m)) * (np.sqrt(0.134 + 0.124 * m))) * r0 / drdx

                TI_QA = round((4.8 * CT ** (0.7) * (TI_amb * 100) ** (0.68) * (dd * D / Xn) ** (-0.57)) / 100, 3)
                TI_tot = round(np.sqrt(np.sum([np.square(TI_amb), np.square(TI_QA)])), 3)

                df = cross_plane.df
                y_grid = np.array(df['x1'])
                z_grid = np.array(df['x2'])
                u_grid = np.array(df['u'])

                n = int(math.sqrt(u_grid.shape[0]))
                Y = y_grid.reshape(n, n)
                Z = z_grid.reshape(n, n)
                U = u_grid.reshape(n, n)

                U_new = flatten_wind_field(Z, U, alfa, Vm, href)
                U_squared, Y_squared, Z_squared, y_c, z_c, flag_no_deficit = get_U_squared(U_new, Y, Z)
                u_max = abs(np.amin(U_squared))

                # in order to account for field with negligible wake deficits, otherwise error in the fitting function
                if flag_no_deficit:
                    A = 0
                    sigma = 0
                    y_gc = 0
                    z_gc = 0
                    err_max = np.amin(abs(U_new))
                    err_mean = 0
                else:
                    u_max = abs(np.amin(U_squared))

                    if u_max < 0.38:
                        # if maximum wake deficit is too small, the fitting function does not work well so it is scaled up,
                        # fitted and then the peak of the gaussian is scaled back down again
                        U_squared = np.multiply(U_squared, 0.38/u_max)
                        popt, perr, yz_arr, u_arr = gauss_interp_2d(Y_squared, Z_squared, y_c, z_c, U_squared)
                        popt[0] = popt[0] * u_max/0.38
                        u_arr = np.divide(u_arr, 0.38/u_max)
                    else:
                        popt, perr, yz_arr, u_arr = gauss_interp_2d(Y_squared, Z_squared, y_c, z_c, U_squared)

                A, y_gc, z_gc, sigmay, sigmaz = popt

                A = round(A, 3) #peak of the gaussian function
                y_gc = round(y_gc, 3) #lateral position of the wake center
                z_gc = round(z_gc, 3) #vertical position of the wake center
                sigmay = round(sigmay, 3) #standard deviation of the gaussian function along the horizontal direction
                sigmaz = round(sigmaz, 3) #standard deviation of the gaussian function along the vertical direction

                # deviation of the gaussian function from the sample points
                interp_values = np.array(gauss_2d(yz_arr, *popt))
                dev = np.abs(np.add(interp_values, -abs(u_arr)))
                err_max = round(np.amax(dev), 3)
                err_mean = round(np.mean(dev), 3)


                for i_o, offset in enumerate(offset_arr):
                    # coordinates of the gaussian center in the reference system centered in the downstream turbine
                    y_c_d = round(y_c[0] + y_gc - offset * D, 3)
                    z_c_d = round(z_c[0] + z_gc, 3)
                    popti = np.array([der, Vm, TI_amb, yaw_angle, dd, offset, A, y_c_d, z_c_d, sigmay, sigmaz, Vm, CT,
                                      TI_QA, TI_tot, err_max, err_mean])
                    gauss_parameters[i_der * m_der + i_v * m_v + i_y * m_y + i_dd * m_dd + i_o, :] = popti


wake_properties = pd.DataFrame(gauss_parameters, columns=['der', 'Vm', 'TI', 'yaw', 'DD', 'offset', 'peak', 'yc_d', 'zc_d',
                                                     'sigmay', 'sigmaz', 'Vm', 'CT', 'TI_QA', 'TI_tot', 'e_max', 'e_mean'])

#save wake properties as a csv to this path
filepath = Path('wake_elliptical.csv')
'''

filepath.parent.mkdir(parents=True, exist_ok=True)
wake_properties.to_csv(filepath, index=False)

#and if I add some other stuff here