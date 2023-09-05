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


'''
#binning considering the lower bound for each parameter
#n° of values, NOT bins (bins=N-1)
n_zc = 4
n_yc = 11
n_A = 11
n_sigma = 11
#n_TI = 3

start_time = time.time()


Vm_bins = np.linspace(3, 15, 7, endpoint=True)
#TI_bins = np.linspace(0.085, 0.155, n_TI, endpoint=True)       #0.085-0.155
sigma_bins = np.around(np.concatenate(([0], np.linspace(86.5, 150.93, n_sigma-1, endpoint=True))), 3)      #sigma_bins = np.linspace(0, 174.73, n_sigma, endpoint=True)     #0-174.73
zc_bins = np.around(np.linspace(127.96, 149.18, n_zc, endpoint=True), 3)     #123.42-164.23
yc_bins = np.around(np.linspace(-142.83, 140.67, n_yc, endpoint=True), 3)     #-148.3-146.12
A_bins = np.around(np.linspace(0, 1.553, n_A, endpoint=True), 3)              #0-3.16


dim_Vm = len(Vm_bins)
#dim_TI = len(TI_bins) - 1
dim_sigma = len(sigma_bins) - 1
dim_zc = len(zc_bins) - 1
dim_yc = len(yc_bins) - 1
dim_A = len(A_bins) - 1


cases = np.zeros((dim_Vm, dim_A, dim_yc, dim_zc, dim_sigma))

df = pd.read_csv(
        'C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/csv backup/wake_5D.csv') #C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/csv backup/wake_circular_noz0.csv


for i in range(0, 875):
    A, yc, zc, sigma, Vm = df.iloc[i][6:11]

    idx_Vm = np.where(Vm_bins == Vm)
    idx_peak = get_lower_bound_idx(A_bins, A)
    idx_yc = get_lower_bound_idx(yc_bins, yc)
    idx_zc = get_lower_bound_idx(zc_bins, zc)
    idx_sigma = get_lower_bound_idx(sigma_bins, sigma)
    #idx_TI = get_lower_bound_idx(TI_bins, TI_add)

    cases[idx_Vm, idx_peak, idx_yc, idx_zc, idx_sigma] = cases[idx_Vm, idx_peak, idx_yc, idx_zc, idx_sigma] + 1

cases_with_index = np.zeros((dim_Vm * dim_A * dim_yc * dim_zc * dim_sigma, 6))

m_Vm = dim_A * dim_yc * dim_zc * dim_sigma #* dim_TI
m_A = dim_yc * dim_zc * dim_sigma #* dim_TI
m_yc = dim_zc * dim_sigma #* dim_TI
m_zc = dim_sigma #* dim_TI
#m_sigma = dim_TI


for idx_Vm in range(0, dim_Vm):
    for idx_peak in range(0, dim_A):
        for idx_yc in range(0, dim_yc):
            for idx_zc in range(0, dim_zc):
                for idx_sigma in range(0, dim_sigma):
                    #for idx_TI in range(0, dim_TI):
                    cases_with_index[
                    m_Vm * idx_Vm + m_A * idx_peak + m_yc * idx_yc + m_zc * idx_zc + idx_sigma, :] \
                        = [idx_Vm, idx_peak, idx_yc, idx_zc, idx_sigma,
                                        cases[idx_Vm, idx_peak, idx_yc, idx_zc, idx_sigma]]

pippaperino = pd.DataFrame(cases_with_index,
                           columns=['idx_Vm', 'idx_peak', 'idx_yc', 'idx_zc', 'idx_sigma',
                                    'occurrences'])
pippaperino.sort_values(by=['occurrences'], ascending=False, inplace=True)


filepath = Path('C:/Users/randr/Desktop/binned_5D.csv')


filepath.parent.mkdir(parents=True, exist_ok=True)
pippaperino.to_csv(filepath, index=False)

occurrences_lim = np.amin(np.where(pippaperino['occurrences'] == 0))

# pippaperino = pd.read_csv('C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/Binning/noz0/sorted_binned_elliptical_noz0.csv')
simulations = np.zeros((occurrences_lim * 16, 5))

for i in range(0, occurrences_lim):
    idx_Vm, idx_peak, idx_yc, idx_zc, idx_sigma = pippaperino.iloc[i][0:5]
    Vm_arr = [Vm_bins[int(idx_Vm)]]
    A_arr = A_bins[int(idx_peak):int(idx_peak) + 2]
    yc_arr = yc_bins[int(idx_yc):int(idx_yc) + 2]
    zc_arr = zc_bins[int(idx_zc):int(idx_zc) + 2]
    sigma_arr = sigma_bins[int(idx_sigma):int(idx_sigma) + 2]
    #TI_arr = TI_bins[int(idx_TI):int(idx_TI) + 2]
    res = [[Vm, A, yc, zc, sigma] for Vm in Vm_arr for A in A_arr for yc in yc_arr for zc in zc_arr for
           sigma in sigma_arr]

    for j in range(0, 16):
        simulations[i * 16 + j, :] = res[j]

plutopino = pd.DataFrame(simulations, columns=['Vm', 'A', 'yc', 'zc', 'sigma'])
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
n_yc = 11
n_A = 11
n_sigma = 11

start_time = time.time()

#arrays of extremes of the bins given the range of each parameter and the n° of values chosen
sigma_values = np.around(np.concatenate(([0], np.linspace(86.5, 150.93, n_sigma-1, endpoint=True))), 3)      #sigma_bins = np.linspace(0, 174.73, n_sigma, endpoint=True)     #0-174.73
zc_values = np.around(np.linspace(127.96, 149.18, n_zc, endpoint=True), 3)     #123.42-164.23
yc_values = np.around(np.linspace(-142.83, 140.67, n_yc, endpoint=True), 3)     #-148.3-146.12
A_values = np.around(np.linspace(0, 1.553, n_A, endpoint=True), 3)              #0-3.16

#arrays of mean values for each bin
Vm_bins = np.linspace(3, 15, 7, endpoint=True)
sigma_bins = np.concatenate(([0], np.ravel([[np.mean(sigma_values[i:i+2])] for i in range(1, len(sigma_values) - 1)])))
zc_bins = np.ravel([[np.mean(zc_values[i:i+2])] for i in range(0, len(zc_values) - 1)])
yc_bins = np.ravel([[np.mean(yc_values[i:i+2])] for i in range(0, len(yc_values) - 1)])
A_bins = np.ravel([[np.mean(A_values[i:i+2])] for i in range(0, len(A_values) - 1)])


dim_Vm = len(Vm_bins)
dim_sigma = len(sigma_bins)
dim_zc = len(zc_bins)
dim_yc = len(yc_bins)
dim_A = len(A_bins)


df = pd.read_csv('C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/csv backup/wake_5D.csv')

#create a matrix to store n° of occurences for each combination of parameters
cases_with_index = np.zeros((dim_Vm * dim_A * dim_yc * dim_zc * dim_sigma, 6))
m_Vm = dim_A * dim_yc * dim_zc * dim_sigma
m_A = dim_yc * dim_zc * dim_sigma
m_yc = dim_zc * dim_sigma
m_zc = dim_sigma

for i in range(0, 875):
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


filepath = Path('C:/Users/randr/Desktop/binned_5D.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
cases_with_binning.to_csv(filepath, index=False)

occurrences_lim = np.amin(np.where(cases_with_binning['occurrences'] == 0))

print(cases_with_binning)
print('The number of significant cases is: ' + str(occurrences_lim))

stop_time = time.time()
time_elapsed = stop_time-start_time
print(time_elapsed)