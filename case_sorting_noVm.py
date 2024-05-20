import numpy as np
import pandas as pd
from pathlib import Path

def get_lower_bound_idx(arr, value):
    idx_lb = np.amax(np.where(value >= arr))
    return idx_lb

def find_nearest(arr, value):
    idx = (np.abs(arr - value)).argmin()
    return idx


df = pd.read_csv('wake_5D.csv')
n_cases = len(df.index)

#min and max of the wake properties
peak_min = df['peak'].min()
peak_max = df['peak'].max() + 0.01
Vm_min = df['Vm'].min()
Vm_max = df['Vm'].max()
sigma_min = df['sigma'].min()
sigma_max = df['sigma'].max() + 0.1
yc_min = df['yc_d'].min()
yc_max = df['yc_d'].max() + 1

#binning considering the lower bound for each parameter
#n° of values that define the interpolation grid, NOT bins (bins=N-1)
n_yc = 5
n_A = 5
n_sigma = 5

#definition of the values of the interpolation grid, zc variations are not considered
Vm_values = np.concatenate(([Vm_min], np.linspace(5, Vm_max, 6, endpoint=True)))
sigma_values = np.around(np.linspace(sigma_min, sigma_max, n_sigma, endpoint=True), 3)
yc_values = np.around(np.linspace(yc_min, yc_max, n_yc, endpoint=True), 3)
A_values = np.around(np.linspace(peak_min, peak_max, n_A, endpoint=True), 3)

#n° of bins for each dimensions, for Vm = n° of values since interpolation over Vm was not required
dim_Vm = len(Vm_values)
dim_sigma = len(sigma_values) - 1
dim_yc = len(yc_values) - 1
dim_A = len(A_values) - 1


cases = np.zeros((dim_Vm, dim_A, dim_yc, dim_sigma))
cases_with_index = np.zeros((dim_Vm * dim_A * dim_yc * dim_sigma, 5))


m_Vm = dim_A * dim_yc * dim_sigma
m_A = dim_yc * dim_sigma
m_yc = dim_sigma


df = df.sort_values(by=['Vm'], ascending=True)

#iteration over the cases in the csv in order to bin and count them
for i in range(0, n_cases):
    A, yc, zc, sigma, Vm = df.iloc[i][6:11]
    idx_Vm = np.where(Vm_values == Vm)[0][0]
    idx_peak = get_lower_bound_idx(A_values, A)
    idx_yc = get_lower_bound_idx(yc_values, yc)
    idx_sigma = get_lower_bound_idx(sigma_values, sigma)
    cases_with_index[m_Vm * idx_Vm + m_A * idx_peak + m_yc * idx_yc + idx_sigma, :] \
        = [idx_Vm, idx_peak, idx_yc, idx_sigma,
           cases_with_index[m_Vm * idx_Vm + m_A * idx_peak + m_yc * idx_yc + idx_sigma, 4] + 1]


cases_with_binning = pd.DataFrame(cases_with_index,
                           columns=['idx_Vm', 'idx_peak', 'idx_yc', 'idx_sigma',
                                    'occurrences'])
cases_with_binning.sort_values(by=['occurrences'], ascending=False, inplace=True)


filepath = Path('binned_5D.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
cases_with_binning.to_csv(filepath, index=False)

try:
    occurrences_lim = np.amin(np.where(cases_with_binning['occurrences'] == 0))
except:
    occurrences_lim = cases_with_binning.shape[0]

simulations = np.zeros((occurrences_lim * 8, 4))

#for each initial case, 8 significant cases are considered which are all the possible combinations of the extremes that
#delimit the initial case's wake parameters
for i in range(0, occurrences_lim):
    idx_Vm, idx_peak, idx_yc, idx_sigma = cases_with_binning.iloc[i][0:4]
    Vm_arr = [Vm_values[int(idx_Vm)]]
    A_arr = A_values[int(idx_peak):int(idx_peak) + 2]
    yc_arr = yc_values[int(idx_yc):int(idx_yc) + 2]
    sigma_arr = sigma_values[int(idx_sigma):int(idx_sigma) + 2]
    res = [[Vm, A, yc, sigma] for Vm in Vm_arr for A in A_arr for yc in yc_arr for sigma in sigma_arr]

    for j in range(0, 8):
        simulations[i * 8 + j, :] = res[j]

significant_cases = pd.DataFrame(simulations, columns=['Vm', 'A', 'yc', 'sigma'])
significant_cases.drop_duplicates(keep='first', inplace=True)

filepath = Path('uniquesim_5D.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
significant_cases.to_csv(filepath, index=False)

n_significant = len(significant_cases.index)
print('The number of significant cases is: ' + str(n_significant))


#what follows is a check to ensure that no significant cases are forgotten
flag = 0

for i in range(0, n_cases):
    A, yc, zc, sigma, Vm = df.iloc[i][6:11]

    idx_Vm = np.where(Vm_values == Vm)
    idx_peak = get_lower_bound_idx(A_values, A)
    idx_yc = get_lower_bound_idx(yc_values, yc)
    idx_sigma = get_lower_bound_idx(sigma_values, sigma)

    A_arr = [A_values[idx_peak], A_values[idx_peak + 1]]
    yc_arr = [yc_values[idx_yc], yc_values[idx_yc + 1]]
    sigma_arr = [sigma_values[idx_sigma], sigma_values[idx_sigma + 1]]

    combs = [[Vm, A_c, yc_c, sigma_c] for A_c in A_arr for yc_c in yc_arr for sigma_c in sigma_arr]


    for j in range(0, 8):
        if not((significant_cases == combs[j]).all(1).any()):
            flag = 1
            exit()



if flag:
    print('Not all cases are represented')
else:
    print('All good, all cases are represented')