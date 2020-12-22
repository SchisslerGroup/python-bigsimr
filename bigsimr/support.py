from scipy.stats import rankdata, kendalltau
import numpy as np
import pandas as pd
import multiprocessing as mp

# Row as variable
def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

def cal_kendall(X):
    return [kendalltau(X[:,0], X[:,j])[0] for j in range(X.shape[1])]

def cor_fast(x, y=None, method="pearson", ncores=4):
    methods = ['pearson', 'spearman', 'kendall']
    if method not in methods:
        raise ValueError("Invalid method. Expected one of: %s" % methods)

    if method == "pearson":
        if y is None:
            return corr2_coeff(x.T, x.T)
        else:
            return np.corrcoef(x, y)[0,1]
    if method == "spearman":
        if y is None:
            x = rankdata(x, axis=0)
            return corr2_coeff(x.T, x.T)
        else:
            x = rankdata(x)
            y = rankdata(y)
            return np.corrcoef(x, y)[0,1]
    if method == "kendall":
        if y is None:
            res = np.empty((x.shape[1], x.shape[1]))
            pool = mp.Pool(processes=ncores)
            results = [pool.apply_async(cal_kendall, args=(x[:,i:],)) for i in range(x.shape[1])]
            pool.close()
            results = [p.get() for p in results]
            for i, result in enumerate(results):
                res[i,i:] = result
                res[i:,i] = result

            # x = pd.DataFrame(x)
            # res = x.corr(method='kendall').values

            return res
        else:
            return kendalltau(x, y)[0]


def cor_convert(rho, case):
    switch = {
        'spearman_pearson': lambda x: 2*np.sin(x*np.pi/6),
        'kendall_pearson': lambda x: np.sin(x*np.pi/2),
        'pearson_spearman': lambda x: (6/np.pi)*np.arcsin(x/2),
        'kendall_spearman': lambda x: (6 / np.pi) * np.arcsin(np.sin(x * np.pi / 2) / 2),
        'pearson_kendall': lambda x: (2/np.pi)*np.arcsin(x),
        'spearman_kendall': lambda x: (2/np.pi)*np.arcsin(2*np.sin(x*np.pi/6))
    }
    if case not in switch:
        raise ValueError("Invalid conversion case. Expected one of: %s" % list(switch))

    return switch[case](rho)

from itertools import combinations

def cor_bounds(margins, type="pearson",cores=1,reps=1e5):
    types = ['pearson', 'spearman', 'kendall']
    if type not in types:
        raise ValueError("Invalid type. Expected one of: %s" % types)

    d = len(margins)
    reps = int(reps)
    index_mat = list(combinations(range(d), 2))

    sim_data = np.empty((reps, d))
    for i in range(d):
        sim_data[:,i] = np.sort(margins[i].rvs(size=reps))

    rho_upper = cor_fast(sim_data, method=type)

    rho_lower = np.zeros((d,d))
    np.fill_diagonal(rho_lower, 0.5)
    for (i, j) in index_mat:
        rho_lower[i,j] = cor_fast(sim_data[:,i], np.flip(sim_data[:,j]), method=type)
    rho_lower = rho_lower + rho_lower.T
    return rho_lower, rho_upper