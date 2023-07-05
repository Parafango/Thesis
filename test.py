import math

from floris.tools import FlorisInterface
import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
import pandas as pd
from floris.tools.visualization import calculate_horizontal_plane_with_turbines
import floris.tools.visualization as wakeviz
from floris.tools.visualization import visualize_cut_plane
from floris.tools.visualization import plot_turbines_with_fi

def I_ishihara(D, H, I0, CT, x, y, z):
    r = np.sqrt(y**2+(z-H)**2)
    k = 0.11*CT**1.07*I0**0.20
    eps = 0.23*CT**(-0.25)*I0**0.17
    sigma = D * (k * x / D + eps)
    d = 2.3*CT**(-1.2)
    e = 1*I0**0.1
    f = 0.7*CT**(-3.2)*I0**(-0.45)
    if r/D <= 0.5:
        k1 = (np.cos(np.pi / 2 * (r / D - 0.5))) ** 2
        k2 = (np.cos(np.pi / 2 * (r / D + 0.5))) ** 2
    if r/D > 0.5:
        k1 = 1
        k2 = 0
    if z >= H:
        delta = 0
    else:
        delta = I0 * (np.sin(np.pi*(H-z)/H))**2
    I_add = (k1 * np.exp(-((r-D/2)**2)/(2*sigma**2)) + k2 * np.exp(-((r+D/2)**2)/(2*sigma**2)))/(d + e * x/D + f * (1 + x/D)**(-2)) - delta
    return I_add

def I_ishihara_1D(D, I0, CT, x):
    d = 2.3*CT**(-1.2)
    e = 1*I0**0.1
    f = 0.7*CT**(-3.2)*I0**(-0.45)
    I_add = 1/(d + e * x/D + f * (1 + x/D)**(-2))
    return I_add

def I_tianson_1D(D, I0, CT, x):
    k = 0.5
    I_add = k/2 * CT**(k/4) * I0**(-k/8) * (x/D)**(-k)
    return I_add

D = 178.3
H = 119
I0 = 0.2
CT = 0.5
z_arr = np.linspace(19, 219, 25)
y_arr = np.linspace(-100, 100, 25)
Y, Z = np.meshgrid(y_arr, z_arr)

Iadd = np.zeros((25,25))

for i in range(0,25):
    for j in range(0,25):
        Iadd[i,j] = I_ishihara(D, H, I0, CT, 5*D, y_arr[j], z_arr[i])


Imax = I_ishihara_1D(D, I0, CT, 5*D)
print(Imax)
Imax = I_tianson_1D(D, I0, CT, 5*D)
print(Imax)

'''
fi = FlorisInterface("Configurations/gch.yaml")

D = fi.floris.farm.rotor_diameters[0]

layout_x = [0., 5*D]
layout_y = [0., 0]

fi.reinitialize(layout_x=layout_x, layout_y=layout_y)
yaw_angles = np.zeros((1,1,2))
yaw_angles[0,0,0] = 15

fi.calculate_wake(yaw_angles=yaw_angles)


fig, axarr = plt.subplots(4,1,sharex=True)

h_plane = fi.calculate_horizontal_plane(height=90,yaw_angles=yaw_angles)
visualize_cut_plane(h_plane, ax=axarr[0], title='15° upstream yaw')
plot_turbines_with_fi(fi, ax=axarr[0],color='k')

yaw_angles[0,0,0] = 15
fi.reinitialize(wind_directions=[285])
fi.calculate_wake(yaw_angles=yaw_angles)

h_plane = fi.calculate_horizontal_plane(height=90,yaw_angles=yaw_angles)
visualize_cut_plane(h_plane, ax=axarr[1], title='15° upstream misalignment')
plot_turbines_with_fi(fi, ax=axarr[1],color='k')

layout_x=[0., 5*D*math.cos(15*math.pi/180)]
layout_y=[0., -5*D*math.sin(15*math.pi/180)]

yaw_angles[0,0,0] = 15
fi.reinitialize(wind_directions=[285], layout_y=layout_y, layout_x=layout_x)
fi.calculate_wake(yaw_angles=yaw_angles)

h_plane = fi.calculate_horizontal_plane(height=90,yaw_angles=yaw_angles)
visualize_cut_plane(h_plane, ax=axarr[2], title='15° yaw upstream + change layout')
plot_turbines_with_fi(fi, ax=axarr[2],color='k')

layout_x=[0., 7*D]
layout_y=[0., 0]

yaw_angles[0,0,0] = 0
fi.reinitialize(wind_directions=[270], layout_y=layout_y, layout_x=layout_x)
fi.calculate_wake(yaw_angles=yaw_angles)

h_plane = fi.calculate_horizontal_plane(height=90,yaw_angles=yaw_angles)
visualize_cut_plane(h_plane, ax=axarr[3], title='270° no yaw')
plot_turbines_with_fi(fi, ax=axarr[3],color='k')

wakeviz.show_plots()

fig, axarr = plt.subplots(3,1,sharex=True)
crossplane1 = fi.calculate_cross_plane(3 * D)
visualize_cut_plane(crossplane1, ax=axarr[0], title='3D')

crossplane2 = fi.calculate_cross_plane(5 * D)
visualize_cut_plane(crossplane2, ax=axarr[1], title='5D')

crossplane3 = fi.calculate_cross_plane(7 * D)
visualize_cut_plane(crossplane3, ax=axarr[2], title='7D')

wakeviz.show_plots()
'''