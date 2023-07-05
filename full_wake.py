import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from floris.tools import FlorisInterface
import math
from matplotlib import cm

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

Vm = 11
DD = 5
der = 0
TI = 0.1
fi.reinitialize(solver_settings=solver_settings, reference_wind_height=href)
fi.reinitialize(layout_x=layout_x, layout_y=layout_y)
fi.reinitialize(wind_speeds=[Vm])
fi.reinitialize(turbulence_intensity=TI)
Vref = fi.floris.flow_field.wind_speeds[0]

yaw_angle = 0
yaw_angles = np.zeros((1,1,1))
yaw_angles[0,0,0] = yaw_angle


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
v_grid = np.array(df['v'])
w_grid = np.array(df['w'])
V_grid = np.array(np.sqrt(np.add(np.square(u_grid), np.square(v_grid), np.square(w_grid))))

n = int(math.sqrt(u_grid.shape[0]))
Y_mat = y_grid.reshape(n, n)
Z_mat = z_grid.reshape(n, n)
u_mat = u_grid.reshape(n, n)
v_mat = v_grid.reshape(n, n)
w_mat = w_grid.reshape(n, n)
V_mat = V_grid.reshape(n, n)

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1,2,1, projection = '3d')
surf = ax.plot_surface(Y_mat, Z_mat, v_mat, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf)

ax.title.set_text('v field')

ax = fig.add_subplot(1,2,2, projection = '3d')
ax.title.set_text('v field')

surf2 = ax.plot_surface(Y_mat, Z_mat, v_mat, cmap=cm.coolwarm,
                             linewidth=0, antialiased = False)

fig.colorbar(surf2)

plt.show()

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1,2,1, projection = '3d')
ax.title.set_text('w field')

surf3 = ax.plot_surface(Y_mat, Z_mat, w_mat, cmap=cm.coolwarm,
                             linewidth=0, antialiased = False)

fig.colorbar(surf3)

ax = fig.add_subplot(1,2,2, projection = '3d')
ax.title.set_text('w field')

surf4 = ax.plot_surface(Y_mat, Z_mat, w_mat, cmap=cm.coolwarm,
                             linewidth=0, antialiased = False)

fig.colorbar(surf4)


plt.show()