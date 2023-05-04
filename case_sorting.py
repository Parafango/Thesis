import math
import numpy as np
import pandas as pd
import time
from pathlib import Path

def get_lower_bound_idx(arr, value):
    idx_lb = np.amax(np.where(value >= arr))
    return idx_lb

def get_par_spaced(a, b, arr_len):
    m = np.mean([a,b])
    bins = np.linspace(np.square(a), np.square(m), int((arr_len+1) / 2))
    bins = np.sqrt(bins)
    bins = np.concatenate((bins, np.add(m, np.abs(np.add(np.flip(bins[0:len(bins) - 1]), -m)))))
    return bins

flag_shape = 'ell' #'circ'
flag_spacing = 'lin' #par
flag_noz0 = True

start_time = time.time()

if flag_shape == 'ell':
    Vm_bins = np.linspace(3, 15, 7, endpoint=True)
    if flag_spacing == 'lin':
        sigmay_bins = np.linspace(33.2, 89.93, 4, endpoint=True)
        sigmaz_bins = np.linspace(33.2, 90.37, 4, endpoint=True)
        zc_bins = np.linspace(120, 130, 3, endpoint=True)
        yc_bins = np.linspace(-202, 199, 4, endpoint=True)
        A_bins = np.linspace(0.31, 7.6, 4, endpoint=True)
    elif flag_spacing == 'par':
        sigmay_bins = get_par_spaced(33.2, 89.93, 5)
        sigmaz_bins = get_par_spaced(33.2, 90.37, 5)
        zc_bins = np.linspace(120, 130, 3, endpoint=True)
        yc_bins = get_par_spaced(-202, 199, 5)
        A_bins = np.linspace(0.31, 7.6, 4, endpoint=True)

    '''
    Vm_bins = np.array([3, 5, 7, 9, 11, 13, 15])
    sigmay_bins = np.array([33.2, 38.95, 44.61, 50.28, 55.94, 61.60, 67.26, 72.93, 78.59, 84.25, 89.93])
    sigmaz_bins = np.array([34.1, 39.73, 45.36, 50.98, 56.61, 62.23, 67.86, 73.48, 79.11, 84.73, 90.37])
    zc_bins = np.array([119, 121.5, 126.5, 131])
    yc_bins = np.array([-202, -161, -121, -82, -42, -2, 38, 78, 118, 158, 199])
    A_bins = np.array([0.31, 1.05, 1.78, 2.5, 3.23, 3.96, 4.68, 5.41, 6.14, 6.87, 7.6])
    '''

    dim_Vm = len(Vm_bins) - 1
    dim_sigmay = len(sigmay_bins) - 1
    dim_sigmaz = len(sigmaz_bins) - 1
    dim_zc = len(zc_bins) - 1
    dim_yc = len(yc_bins) - 1
    dim_A = len(A_bins) - 1

    cases = np.zeros((dim_Vm, dim_A, dim_yc, dim_zc, dim_sigmay, dim_sigmaz))
    if flag_noz0:
        df = pd.read_csv('C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/u/csv backup/new_Vm/wake_elliptical_noz0.csv')
    else:
        df = pd.read_csv('C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/u/csv backup/new_Vm/wake_elliptical.csv')

    for i in range(0, 13125):
        A, yc, zc, sigmay, sigmaz, Vm = df.iloc[i][6:12]

        if Vm == 15:
            idx_Vm = get_lower_bound_idx(Vm_bins, Vm) - 1   #lump all of the Vm=15 cases in the 14-15 bin (all of the other ws have only 1 bin)
        else:
            idx_Vm = get_lower_bound_idx(Vm_bins, Vm)

        idx_peak = get_lower_bound_idx(A_bins, A)
        idx_yc = get_lower_bound_idx(yc_bins, yc)
        idx_zc = get_lower_bound_idx(zc_bins, zc)
        idx_sigmay = get_lower_bound_idx(sigmay_bins, sigmay)
        idx_sigmaz = get_lower_bound_idx(sigmaz_bins, sigmaz)

        cases[idx_Vm, idx_peak, idx_yc, idx_zc, idx_sigmay, idx_sigmaz] = cases[idx_Vm, idx_peak, idx_yc, idx_zc, idx_sigmay, idx_sigmaz] + 1

    cases_with_index = np.zeros((dim_Vm * dim_A * dim_yc * dim_zc * dim_sigmay * dim_sigmaz, 7))

    m_Vm = dim_A * dim_yc * dim_zc * dim_sigmay * dim_sigmaz
    m_A = dim_yc * dim_zc * dim_sigmay * dim_sigmaz
    m_yc = dim_zc * dim_sigmay * dim_sigmaz
    m_zc = dim_sigmay * dim_sigmaz
    m_sigmay = dim_sigmaz

    for idx_Vm in range(0, dim_Vm):
        for idx_peak in range(0, dim_A):
            for idx_yc in range(0, dim_yc):
                for idx_zc in range(0, dim_zc):
                    for idx_sigmay in range(0, dim_sigmay):
                        for idx_sigmaz in range(0, dim_sigmaz):
                            cases_with_index[m_Vm * idx_Vm + m_A * idx_peak + m_yc * idx_yc + m_zc * idx_zc + m_sigmay * idx_sigmay
                            + idx_sigmaz, :] = [idx_Vm, idx_peak, idx_yc , idx_zc, idx_sigmay, idx_sigmaz,
                                                cases[idx_Vm, idx_peak, idx_yc, idx_zc, idx_sigmay, idx_sigmaz]]


    pippaperino = pd.DataFrame(cases_with_index, columns=['idx_Vm', 'idx_peak', 'idx_yc', 'idx_zc', 'idx_sigmay', 'idx_sigmaz', 'occurrences'])
    pippaperino.sort_values(by=['occurrences'], ascending=False, inplace=True)

    if flag_noz0:
        filepath = Path('C:/Users/randr/Desktop/binned_elliptical_noz0.csv')
    else:
        filepath = Path('C:/Users/randr/Desktop/binned_elliptical.csv')

    filepath.parent.mkdir(parents=True, exist_ok=True)
    pippaperino.to_csv(filepath, index=False)

    occurrences_lim = np.amin(np.where(pippaperino['occurrences'] == 0))

    #pippaperino = pd.read_csv('C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/Binning/noz0/sorted_binned_elliptical_noz0.csv')
    simulations = np.zeros((occurrences_lim * 64, 6))

    for i in range(0,occurrences_lim):
        idx_Vm, idx_peak, idx_yc, idx_zc, idx_sigmay, idx_sigmaz = pippaperino.iloc[i][0:6]
        Vm_arr = Vm_bins[int(idx_Vm):int(idx_Vm) + 2]
        A_arr = A_bins[int(idx_peak):int(idx_peak) + 2]
        yc_arr = yc_bins[int(idx_yc):int(idx_yc) + 2]
        zc_arr = zc_bins[int(idx_zc):int(idx_zc) + 2]
        sigmay_arr = sigmay_bins[int(idx_sigmay):int(idx_sigmay) + 2]
        sigmaz_arr = sigmaz_bins[int(idx_sigmaz):int(idx_sigmaz) + 2]
        res = [[Vm, A, yc, zc, sigmay, sigmaz]  for Vm in Vm_arr for A in A_arr for yc in yc_arr for zc in zc_arr for sigmay in sigmaz_arr
               for sigmaz in sigmaz_arr]

        for j in range(0,64):
            simulations[i * 64 + j, :] = res[j]

    plutopino = pd.DataFrame(simulations, columns=['Vm', 'A', 'yc', 'zc', 'sigmay', 'sigmaz'])
    plutopino.drop_duplicates(keep='first', inplace=True)
    if flag_noz0:
        filepath = Path('C:/Users/randr/Desktop/uniquesim_elliptical_noz0.csv')
    else:
        filepath = Path('C:/Users/randr/Desktop/uniquesim_elliptical.csv')

    filepath.parent.mkdir(parents=True, exist_ok=True)
    plutopino.to_csv(filepath, index=False)

elif flag_shape == 'circ':
    Vm_bins = np.linspace(3, 15, 7, endpoint=True)
    if flag_spacing == 'lin':
        sigma_bins = np.linspace(33.2, 89.93, 4, endpoint=True)
        zc_bins = np.linspace(120, 130, 2, endpoint=True)
        yc_bins = np.linspace(-204, 199, 4, endpoint=True)
        A_bins = np.linspace(0.31, 7.6, 4, endpoint=True)
    elif flag_spacing == 'par':
        sigma_bins = get_par_spaced(33.2, 89.93, 5)
        zc_bins = np.linspace(120, 130, 3, endpoint=True)
        yc_bins = get_par_spaced(-202, 199, 5)
        A_bins = np.linspace(0.31, 7.6, 4, endpoint=True)


    dim_Vm = len(Vm_bins) - 1
    dim_sigma = len(sigma_bins) - 1
    dim_zc = len(zc_bins) - 1
    dim_yc = len(yc_bins) - 1
    dim_A = len(A_bins) - 1

    cases = np.zeros((dim_Vm, dim_A, dim_yc, dim_zc, dim_sigma))

    if flag_noz0:
        df = pd.read_csv(
            'C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/u/csv backup/new_Vm/wake_circular_noz0.csv')
    else:
        df = pd.read_csv(
            'C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/u/csv backup/new_Vm/wake_circular.csv')

    for i in range(0, 13125):
        A, yc, zc, sigma, Vm = df.iloc[i][6:11]

        if Vm == 15:
            idx_Vm = get_lower_bound_idx(Vm_bins,
                                         Vm) - 1  # lump all of the Vm=15 cases in the 14-15 bin (all of the other ws have only 1 bin)
        else:
            idx_Vm = get_lower_bound_idx(Vm_bins, Vm)

        idx_peak = get_lower_bound_idx(A_bins, A)
        idx_yc = get_lower_bound_idx(yc_bins, yc)
        idx_zc = get_lower_bound_idx(zc_bins, zc)
        idx_sigma = get_lower_bound_idx(sigma_bins, sigma)

        cases[idx_Vm, idx_peak, idx_yc, idx_zc, idx_sigma] = cases[idx_Vm, idx_peak, idx_yc, idx_zc, idx_sigma] + 1

    cases_with_index = np.zeros((dim_Vm * dim_A * dim_yc * dim_zc * dim_sigma, 6))

    m_Vm = dim_A * dim_yc * dim_zc * dim_sigma
    m_A = dim_yc * dim_zc * dim_sigma
    m_yc = dim_zc * dim_sigma
    m_zc = dim_sigma


    for idx_Vm in range(0, dim_Vm):
        for idx_peak in range(0, dim_A):
            for idx_yc in range(0, dim_yc):
                for idx_zc in range(0, dim_zc):
                    for idx_sigma in range(0, dim_sigma):
                        cases_with_index[
                        m_Vm * idx_Vm + m_A * idx_peak + m_yc * idx_yc + m_zc * idx_zc + idx_sigma, :] \
                            = [idx_Vm, idx_peak, idx_yc, idx_zc, idx_sigma,
                                            cases[idx_Vm, idx_peak, idx_yc, idx_zc, idx_sigma]]

    pippaperino = pd.DataFrame(cases_with_index,
                               columns=['idx_Vm', 'idx_peak', 'idx_yc', 'idx_zc', 'idx_sigma',
                                        'occurrences'])
    pippaperino.sort_values(by=['occurrences'], ascending=False, inplace=True)

    if flag_noz0:
        filepath = Path('C:/Users/randr/Desktop/binned_circular_noz0.csv')
    else:
        filepath = Path('C:/Users/randr/Desktop/binned_circular.csv')

    filepath.parent.mkdir(parents=True, exist_ok=True)
    pippaperino.to_csv(filepath, index=False)

    occurrences_lim = np.amin(np.where(pippaperino['occurrences'] == 0))

    # pippaperino = pd.read_csv('C:/Users/randr/OneDrive - Politecnico di Milano/Tesi/Gaussian wake/Binning/noz0/sorted_binned_elliptical_noz0.csv')
    simulations = np.zeros((occurrences_lim * 32, 5))

    for i in range(0, occurrences_lim):
        idx_Vm, idx_peak, idx_yc, idx_zc, idx_sigma = pippaperino.iloc[i][0:5]
        Vm_arr = Vm_bins[int(idx_Vm):int(idx_Vm) + 2]
        A_arr = A_bins[int(idx_peak):int(idx_peak) + 2]
        yc_arr = yc_bins[int(idx_yc):int(idx_yc) + 2]
        zc_arr = zc_bins[int(idx_zc):int(idx_zc) + 2]
        sigma_arr = sigma_bins[int(idx_sigma):int(idx_sigma) + 2]
        res = [[Vm, A, yc, zc, sigma] for Vm in Vm_arr for A in A_arr for yc in yc_arr for zc in zc_arr for
               sigma in sigma_arr]

        for j in range(0, 32):
            simulations[i * 32 + j, :] = res[j]

    plutopino = pd.DataFrame(simulations, columns=['Vm', 'A', 'yc', 'zc', 'sigma'])
    plutopino.drop_duplicates(keep='first', inplace=True)

    if flag_noz0:
        filepath = Path('C:/Users/randr/Desktop/uniquesim_circular_noz0.csv')
    else:
        filepath = Path('C:/Users/randr/Desktop/uniquesim_circular.csv')

    filepath.parent.mkdir(parents=True, exist_ok=True)
    plutopino.to_csv(filepath, index=False)


print(pippaperino)
print(plutopino)

stop_time = time.time()
time_elapsed = stop_time-start_time
print(time_elapsed)
