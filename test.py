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