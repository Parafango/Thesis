import math
import numpy as np
import pandas as pd
import time
from pathlib import Path

def get_lower_bound_idx(arr, value):
    idx_lb = np.amax(np.where(value >= arr))
    return idx_lb

n_zc = 2
n_yc = 5
n_A = 5
n_sigma = 5

Vm_bins = np.concatenate(([4], np.linspace(5, 15, 6, endpoint=True)))
sigma_bins = np.around(np.linspace(86.5, 149.3, n_sigma, endpoint=True), 3)
yc_bins = np.around(np.linspace(-142.83, 140.67, n_yc, endpoint=True), 3)
A_bins = np.around(np.linspace(0.163, 1.553, n_A, endpoint=True), 3)

uniquesim = pd.read_csv('C:/Users/randr/Desktop/uniquesim_5D.csv')
total_cases = pd.read_csv('C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/csv backup/wake_5D_no3ms.csv')

n_tot_cases = total_cases.shape[0]

flag = 0

for i in range(0, n_tot_cases):
    A, yc, zc, sigma, Vm = total_cases.iloc[i][6:11]

    idx_Vm = np.where(Vm_bins == Vm)
    idx_peak = get_lower_bound_idx(A_bins, A)
    idx_yc = get_lower_bound_idx(yc_bins, yc)
    idx_sigma = get_lower_bound_idx(sigma_bins, sigma)

    A_arr = [A_bins[idx_peak], A_bins[idx_peak + 1]]
    yc_arr = [yc_bins[idx_yc], yc_bins[idx_yc + 1]]
    sigma_arr = [sigma_bins[idx_sigma], sigma_bins[idx_sigma + 1]]

    combs = [[Vm, A_c, yc_c, sigma_c] for A_c in A_arr for yc_c in yc_arr for sigma_c in sigma_arr]


    for j in range(0, 8):
        if not((uniquesim == combs[j]).all(1).any()):
            flag = 1
            exit()



if flag:
    print('Not all cases are represented')
else:
    print('All good')

'''
Vm_bins = np.linspace(3, 15, 7, endpoint=True)
sigma_bins = np.asarray([0, 86.5, 107.98, 129.45, 150.93])      #sigma_bins = np.linspace(0, 174.73, n_sigma, endpoint=True)     #0-174.73
zc_bins = np.linspace(127.96, 149.18, n_zc, endpoint=True)      #123.42-164.23
yc_bins = np.linspace(-142.83, 140.67, n_yc, endpoint=True)     #-148.3-146.12
A_bins = np.linspace(0, 1.553, n_A, endpoint=True)

uniquesim = pd.read_csv('C:/Users/randr/Desktop/uniquesim_5D.csv')
total_cases = pd.read_csv('C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/csv backup/wake_5D.csv')

n_tot_cases = total_cases.shape [0]

flag = 0

for i in range(0, n_tot_cases):
    A, yc, zc, sigma, Vm = total_cases.iloc[i][6:11]

    idx_Vm = np.where(Vm_bins == Vm)
    idx_peak = get_lower_bound_idx(A_bins, A)
    idx_yc = get_lower_bound_idx(yc_bins, yc)
    idx_zc = get_lower_bound_idx(zc_bins, zc)
    idx_sigma = get_lower_bound_idx(sigma_bins, sigma)

    A_arr = [A_bins[idx_peak], A_bins[idx_peak + 1]]
    yc_arr = [yc_bins[idx_yc], yc_bins[idx_yc + 1]]
    zc_arr = [zc_bins[idx_zc], zc_bins[idx_zc + 1]]
    sigma_arr = [sigma_bins[idx_sigma], sigma_bins[idx_sigma + 1]]

    combs = [[Vm, A_c, yc_c, zc_c, sigma_c] for A_c in A_arr for yc_c in yc_arr for zc_c in zc_arr for
           sigma_c in sigma_arr]


    for j in range(0, 16):
        if not((uniquesim == combs[j]).all(1).any()):
            flag = 1
            exit()



if flag:
    print('Not all cases are represented')
else:
    print('All good')
'''
