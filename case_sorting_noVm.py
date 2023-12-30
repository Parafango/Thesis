import math
import numpy as np
import pandas as pd
import time
from pathlib import Path

def get_lower_bound_idx(arr, value):
    idx_lb = np.amax(np.where(value >= arr))
    return idx_lb

def find_nearest(arr, value):
    idx = (np.abs(arr - value)).argmin()
    return idx

def find_direction(ref_value, value):
    if value > ref_value:
        return 1
    elif value < ref_value:
        return - 1
    else:
        return 0

def does_extreme_exist(idx, n_bins, dir):
    if idx == 0 and dir == - 1:
        return -1
    elif idx == n_bins - 1 and dir == 1:
        return -1
    else:
        return 0



#binning considering the lower bound for each parameter
#n° of values, NOT bins (bins=N-1)
n_zc = 2
n_yc = 5
n_A = 5
n_sigma = 5

start_time = time.time()


#all Vm
Vm_bins = np.concatenate(([4], np.linspace(5, 15, 6, endpoint=True)))
sigma_bins = np.around(np.linspace(86.5, 149.3, n_sigma, endpoint=True), 3)
yc_bins = np.around(np.linspace(-142.83, 140.67, n_yc, endpoint=True), 3)
A_bins = np.around(np.linspace(0.163, 1.553, n_A, endpoint=True), 3)

print(A_bins)
'''
#Vm=5:11
Vm_bins = np.linspace(5, 11, 4, endpoint=True)
sigma_bins = np.around(np.linspace(98.86, 135.58, n_sigma, endpoint=True), 3)
yc_bins = np.around(np.linspace(-142.83, 140.67, n_yc, endpoint=True), 3)
A_bins = np.around(np.linspace(0.254, 1.553, n_A, endpoint=True), 3)


#Vm=5:15
Vm_bins = np.linspace(5, 15, 6, endpoint=True)
sigma_bins = np.around(np.linspace(86.5, 135.58, n_sigma, endpoint=True), 3)
yc_bins = np.around(np.linspace(-142.83, 140.67, n_yc, endpoint=True), 3)
A_bins = np.around(np.linspace(0.254, 1.553, n_A, endpoint=True), 3)
'''

dim_Vm = len(Vm_bins)
dim_sigma = len(sigma_bins) - 1
dim_yc = len(yc_bins) - 1
dim_A = len(A_bins) - 1


cases = np.zeros((dim_Vm, dim_A, dim_yc, dim_sigma))

df = pd.read_csv(
        'C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/csv backup/wake_5D_no3ms.csv') #C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/csv backup/wake_circular_noz0.csv

df = df.sort_values(by=['Vm'], ascending=True)

for i in range(0, 875):
    A, yc, zc, sigma, Vm = df.iloc[i][6:11]
    idx_Vm = np.where(Vm_bins == Vm)
    idx_peak = get_lower_bound_idx(A_bins, A)
    idx_yc = get_lower_bound_idx(yc_bins, yc)
    idx_sigma = get_lower_bound_idx(sigma_bins, sigma)

    cases[idx_Vm, idx_peak, idx_yc, idx_sigma] = cases[idx_Vm, idx_peak, idx_yc, idx_sigma] + 1

cases_with_index = np.zeros((dim_Vm * dim_A * dim_yc * dim_sigma, 5))

m_Vm = dim_A * dim_yc * dim_sigma
m_A = dim_yc * dim_sigma
m_yc = dim_sigma


for idx_Vm in range(0, dim_Vm):
    for idx_peak in range(0, dim_A):
        for idx_yc in range(0, dim_yc):
            for idx_sigma in range(0, dim_sigma):
                cases_with_index[
                m_Vm * idx_Vm + m_A * idx_peak + m_yc * idx_yc + idx_sigma, :] \
                    = [idx_Vm, idx_peak, idx_yc, idx_sigma,
                                    cases[idx_Vm, idx_peak, idx_yc, idx_sigma]]

pippaperino = pd.DataFrame(cases_with_index,
                           columns=['idx_Vm', 'idx_peak', 'idx_yc', 'idx_sigma',
                                    'occurrences'])
pippaperino.sort_values(by=['occurrences'], ascending=False, inplace=True)


filepath = Path('C:/Users/randr/Desktop/binned_5D.csv')


filepath.parent.mkdir(parents=True, exist_ok=True)
pippaperino.to_csv(filepath, index=False)

occurrences_lim = np.amin(np.where(pippaperino['occurrences'] == 0))

simulations = np.zeros((occurrences_lim * 8, 4))

for i in range(0, occurrences_lim):
    idx_Vm, idx_peak, idx_yc, idx_sigma = pippaperino.iloc[i][0:4]
    Vm_arr = [Vm_bins[int(idx_Vm)]]
    A_arr = A_bins[int(idx_peak):int(idx_peak) + 2]
    yc_arr = yc_bins[int(idx_yc):int(idx_yc) + 2]
    sigma_arr = sigma_bins[int(idx_sigma):int(idx_sigma) + 2]
    res = [[Vm, A, yc, sigma] for Vm in Vm_arr for A in A_arr for yc in yc_arr for sigma in sigma_arr]

    for j in range(0, 8):
        simulations[i * 8 + j, :] = res[j]

plutopino = pd.DataFrame(simulations, columns=['Vm', 'A', 'yc', 'sigma'])
plutopino.drop_duplicates(keep='first', inplace=True)


filepath = Path('C:/Users/randr/Desktop/uniquesim_5D.csv')


filepath.parent.mkdir(parents=True, exist_ok=True)
plutopino.to_csv(filepath, index=False)


print(pippaperino)
print(occurrences_lim)
print(plutopino)

stop_time = time.time()
time_elapsed = stop_time-start_time
print(time_elapsed)



'''
#binning considering the closest bin for each parameter
#n° of values, NOT bins (bins=N-1)
n_zc = 4
n_yc = 10
n_A = 11
n_sigma = 11

start_time = time.time()

#arrays of extremes of the bins given the range of each parameter and the n° of values chosen
#sigma_values = np.around(np.concatenate(([0], np.linspace(86.5, 150.93, n_sigma-1, endpoint=True))), 3)      #sigma_bins = np.linspace(0, 174.73, n_sigma, endpoint=True)     #0-174.73
#zc_values = np.around(np.linspace(127.96, 149.18, n_zc, endpoint=True), 3)     #123.42-164.23
#yc_values = np.around(np.linspace(-142.83, 140.67, n_yc, endpoint=True), 3)     #-148.3-146.12
#A_values = np.around(np.linspace(0, 1.553, n_A, endpoint=True), 3)              #0-3.16

sigma_values = np.around(np.linspace(86.5, 149.3, n_sigma, endpoint=True), 3)      #sigma_bins = np.linspace(0, 174.73, n_sigma, endpoint=True)     #0-174.73
zc_values = np.around(np.linspace(127.96, 145.5, n_zc, endpoint=True), 3)     #123.42-164.23
yc_values = np.around(np.linspace(-142.83, 140.67, n_yc, endpoint=True), 3)     #-148.3-146.12
A_values = np.around(np.linspace(0.163, 1.553, n_A, endpoint=True), 3)              #0-3.16

#arrays of mean values for each bin
Vm_bins = np.concatenate(([4], np.linspace(5, 15, 6, endpoint=True)))
sigma_bins = np.ravel([[np.mean(sigma_values[i:i+2])] for i in range(0, len(sigma_values) - 1)])
zc_bins = np.ravel([[np.mean(zc_values[i:i+2])] for i in range(0, len(zc_values) - 1)])
yc_bins = np.ravel([[np.mean(yc_values[i:i+2])] for i in range(0, len(yc_values) - 1)])
A_bins = np.ravel([[np.mean(A_values[i:i+2])] for i in range(0, len(A_values) - 1)])

A_step = A_bins[1] - A_bins[0]
yc_step = yc_bins[1] - yc_bins[0]
sigma_step = sigma_bins[2] - sigma_bins[1]

if n_zc == 2:
    zc_step = zc_values[1] - zc_values[0]
else:
    zc_step = zc_bins[1] - zc_bins[0]

dim_Vm = len(Vm_bins)
dim_sigma = len(sigma_bins)
dim_zc = len(zc_bins)
dim_yc = len(yc_bins)
dim_A = len(A_bins)


#df = pd.read_csv('C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/csv backup/wake_5D.csv')
df = pd.read_csv('C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/csv backup/wake_5D_no3ms.csv')

n_initial_cases = df.shape[0]

#create a matrix to store n° of occurences for each combination of parameters
cases_with_index = np.zeros((dim_Vm * dim_A * dim_yc * dim_zc * dim_sigma, 6))
m_Vm = dim_A * dim_yc * dim_zc * dim_sigma
m_A = dim_yc * dim_zc * dim_sigma
m_yc = dim_zc * dim_sigma
m_zc = dim_sigma

for i in range(0, n_initial_cases):
    A, yc, zc, sigma, Vm = df.iloc[i][6:11]
    idx_Vm = np.where(Vm_bins == Vm)[0][0]
    idx_peak = find_nearest(A_bins, A)
    idx_yc = find_nearest(yc_bins, yc)
    idx_zc = find_nearest(zc_bins, zc)
    idx_sigma = find_nearest(sigma_bins, sigma)

    #access the case determined by idxs and increase the occurences column by 1 each time those set of idxs are found
    cases_with_index[m_Vm * idx_Vm + m_A * idx_peak + m_yc * idx_yc + m_zc * idx_zc + idx_sigma, :] \
        = [idx_Vm, idx_peak, idx_yc, idx_zc, idx_sigma,
           cases_with_index[m_Vm * idx_Vm + m_A * idx_peak + m_yc * idx_yc + m_zc * idx_zc + idx_sigma, 5] + 1]

#make database and csv file from the matrix
cases_with_binning = pd.DataFrame(cases_with_index,
                           columns=['idx_Vm', 'idx_peak', 'idx_yc', 'idx_zc', 'idx_sigma',
                                    'occurrences'])
cases_with_binning.sort_values(by=['occurrences'], ascending=False, inplace=True)


try:
    occurrences_lim = np.amin(np.where(cases_with_binning['occurrences'] == 0))
except:
    occurrences_lim = cases_with_binning.shape[0]

cases_with_binning = cases_with_binning.head(n=occurrences_lim)

cases_with_binning.sort_values(by=['idx_Vm'], ascending=True, inplace=True)

filepath = Path('C:/Users/randr/Desktop/binned_5D.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
cases_with_binning.to_csv(filepath, index=False)



print(cases_with_binning)
print('The number of significant cases is: ' + str(occurrences_lim))


significant_cases = np.zeros((occurrences_lim, 5))

for j in range(0, occurrences_lim):
    significant_cases[j, :] = [cases_with_binning.iloc[j][0], cases_with_binning.iloc[j][1], cases_with_binning.iloc[j][2], cases_with_binning.iloc[j][3], cases_with_binning.iloc[j][4]]


interp_feasibility = np.zeros((n_initial_cases, 6))

#check if cases for interpolation exist for each initial case
for i in range(0, n_initial_cases):

    A, yc, zc, sigma, Vm = df.iloc[i][6:11]
    idx_Vm = np.where(Vm_bins == Vm)[0][0]
    idx_peak = find_nearest(A_bins, A)
    idx_yc = find_nearest(yc_bins, yc)
    idx_zc = find_nearest(zc_bins, zc)
    idx_sigma = find_nearest(sigma_bins, sigma)

    interp_feasibility[i, 1] = 2 * abs(A - A_bins[idx_peak]) / A_step
    interp_feasibility[i, 2] = 2 * abs(yc - yc_bins[idx_yc]) / yc_step
    interp_feasibility[i, 3] = 2 * abs(zc - zc_bins[idx_zc]) / zc_step
    interp_feasibility[i, 4] = 2 * abs(sigma - sigma_bins[idx_sigma]) / sigma_step
    interp_feasibility[i, 5] = np.sqrt(np.sum(np.square(interp_feasibility[i, 1:5])))

    peak_dir = find_direction(A_bins[idx_peak], A)
    yc_dir = find_direction(yc_bins[idx_yc], yc)
    zc_dir = find_direction(zc_bins[idx_zc], zc)
    sigma_dir = find_direction(sigma_bins[idx_sigma], sigma)

    #check if case is already in the external region where interpolation is not possible if cases are not defined at the extremes
    peak_exist = does_extreme_exist(idx_peak, dim_A, peak_dir)
    yc_exist = does_extreme_exist(idx_yc, dim_yc, yc_dir)
    zc_exist = does_extreme_exist(idx_zc, dim_zc, zc_dir)
    sigma_exist = does_extreme_exist(idx_sigma, dim_sigma, sigma_dir)
    arr_exist = [peak_exist, yc_exist, zc_exist, sigma_exist]


    if any(arr_exist):
        interp_feasibility[i, 0] = 1
        continue

    peak_arr = [idx_peak, idx_peak + peak_dir]
    yc_arr = [idx_yc, idx_yc + yc_dir]
    zc_arr = [idx_zc, idx_zc + zc_dir]
    sigma_arr = [idx_sigma, idx_sigma + sigma_dir]

    break_flag = 0

    for j_peak in peak_arr:
        for j_yc in yc_arr:
            for j_zc in zc_arr:
                for j_sigma in sigma_arr:
                    if not([idx_Vm, j_peak, j_yc, j_zc, j_sigma] in significant_cases.tolist()):
                        interp_feasibility[i, 0] = 1
                        break_flag = 1
                        break
                if break_flag:
                    break
            if break_flag:
                break
        if break_flag:
            break


print('\n')
print(interp_feasibility)
print(sum(interp_feasibility))
print(np.amax(interp_feasibility, axis=0))
print(np.average(interp_feasibility, axis=0))

print('\n')

print('Peak step: ' + str(A_step) + ' m/s')
print('Yc step: ' + str(yc_step) + ' m')
print('Zc step: ' + str(zc_step) + ' m')
print('Sigma step: ' + str(sigma_step) + ' m')

print('\n')

print('Peak step/peak range: ' + str(A_step / (A_values[-1] - A_values[0])))
print('Yc step/Yc range: ' + str(yc_step / (yc_values[-1] - yc_values[0])))
print('Zc step/Zc range: ' + str(zc_step / (zc_values[-1] - zc_values[0])))
print('Sigma step/Sigma range: ' + str(sigma_step / (sigma_values[-1] - sigma_values[1])))




'''
'''
#binning considering the closest bin for each parameter
#n° of values, NOT bins (bins=N-1)
n_zc = 2
n_yc = 6
n_A = 6
n_sigma = 6
n_TI = 6

start_time = time.time()

#arrays of extremes of the bins given the range of each parameter and the n° of values chosen
#sigma_values = np.around(np.concatenate(([0], np.linspace(86.5, 150.93, n_sigma-1, endpoint=True))), 3)      #sigma_bins = np.linspace(0, 174.73, n_sigma, endpoint=True)     #0-174.73
#zc_values = np.around(np.linspace(127.96, 149.18, n_zc, endpoint=True), 3)     #123.42-164.23
#yc_values = np.around(np.linspace(-142.83, 140.67, n_yc, endpoint=True), 3)     #-148.3-146.12
#A_values = np.around(np.linspace(0, 1.553, n_A, endpoint=True), 3)              #0-3.16

sigma_values = np.around(np.linspace(86.5, 149.3, n_sigma, endpoint=True), 3)      #sigma_bins = np.linspace(0, 174.73, n_sigma, endpoint=True)     #0-174.73
zc_values = np.around(np.linspace(127.96, 145.5, n_zc, endpoint=True), 3)     #123.42-164.23
yc_values = np.around(np.linspace(-142.83, 140.67, n_yc, endpoint=True), 3)     #-148.3-146.12
A_values = np.around(np.linspace(0.163, 1.553, n_A, endpoint=True), 3)              #0-3.16
TI_values = np.around(np.linspace(0.188,0.386, n_TI, endpoint=True), 3)

#arrays of mean values for each bin
Vm_bins = np.concatenate(([4], np.linspace(5, 15, 6, endpoint=True)))
sigma_bins = np.ravel([[np.mean(sigma_values[i:i+2])] for i in range(0, len(sigma_values) - 1)])
zc_bins = np.ravel([[np.mean(zc_values[i:i+2])] for i in range(0, len(zc_values) - 1)])
yc_bins = np.ravel([[np.mean(yc_values[i:i+2])] for i in range(0, len(yc_values) - 1)])
A_bins = np.ravel([[np.mean(A_values[i:i+2])] for i in range(0, len(A_values) - 1)])
TI_bins = np.ravel([[np.mean(TI_values[i:i+2])] for i in range(0, len(TI_values) - 1)])

A_step = A_bins[1] - A_bins[0]
yc_step = yc_bins[1] - yc_bins[0]
sigma_step = sigma_bins[2] - sigma_bins[1]
TI_step = TI_bins[2] - TI_bins[1]

if n_zc == 2:
    zc_step = zc_values[1] - zc_values[0]
else:
    zc_step = zc_bins[1] - zc_bins[0]

dim_Vm = len(Vm_bins)
dim_sigma = len(sigma_bins)
dim_zc = len(zc_bins)
dim_yc = len(yc_bins)
dim_A = len(A_bins)
dim_TI = len(TI_bins)


#df = pd.read_csv('C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/csv backup/wake_5D.csv')
df = pd.read_csv('C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/csv backup/wake_5D_no3ms.csv')

n_initial_cases = df.shape[0]

#create a matrix to store n° of occurences for each combination of parameters
cases_with_index = np.zeros((dim_Vm * dim_A * dim_yc * dim_zc * dim_sigma * dim_TI, 7))
m_Vm = dim_A * dim_yc * dim_zc * dim_sigma * dim_TI
m_A = dim_yc * dim_zc * dim_sigma * dim_TI
m_yc = dim_zc * dim_sigma * dim_TI
m_zc = dim_sigma * dim_TI
m_sigma = dim_TI


for i in range(0, n_initial_cases):
    A, yc, zc, sigma, Vm = df.iloc[i][6:11]
    TI_amb = df.iloc[i][2]
    TI_add = df.iloc[i][11]
    TI = np.sqrt(TI_amb**2+TI_add**2)
    idx_Vm = np.where(Vm_bins == Vm)[0][0]
    idx_peak = find_nearest(A_bins, A)
    idx_yc = find_nearest(yc_bins, yc)
    idx_zc = find_nearest(zc_bins, zc)
    idx_sigma = find_nearest(sigma_bins, sigma)
    idx_TI = find_nearest(TI_bins, TI)

    #access the case determined by idxs and increase the occurences column by 1 each time those set of idxs are found
    cases_with_index[m_Vm * idx_Vm + m_A * idx_peak + m_yc * idx_yc + m_zc * idx_zc + m_sigma * idx_sigma + idx_TI, :] \
        = [idx_Vm, idx_peak, idx_yc, idx_zc, idx_sigma, idx_TI,
           cases_with_index[m_Vm * idx_Vm + m_A * idx_peak + m_yc * idx_yc + m_zc * idx_zc + m_sigma * idx_sigma + idx_TI, 6] + 1]

#make database and csv file from the matrix
cases_with_binning = pd.DataFrame(cases_with_index,
                           columns=['idx_Vm', 'idx_peak', 'idx_yc', 'idx_zc', 'idx_sigma', 'idx_TI',
                                    'occurrences'])
cases_with_binning.sort_values(by=['occurrences'], ascending=False, inplace=True)


try:
    occurrences_lim = np.amin(np.where(cases_with_binning['occurrences'] == 0))
except:
    occurrences_lim = cases_with_binning.shape[0]

cases_with_binning = cases_with_binning.head(n=occurrences_lim)

cases_with_binning.sort_values(by=['idx_Vm'], ascending=True, inplace=True)

filepath = Path('C:/Users/randr/Desktop/binned_5D.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
cases_with_binning.to_csv(filepath, index=False)



print(cases_with_binning)
print('The number of significant cases is: ' + str(occurrences_lim))
'''