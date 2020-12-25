import jax
import numpy as np
import scipy
import os
from bigsimr.cor_nearPSD import cor_nearPSD
import multiprocessing as mp


def jax_rmvuu(n, rho):
    R = jax.device_put(np.array(rho))
    m = jax.numpy.zeros(rho.shape[0])

    key = jax.random.PRNGKey(np.random.randint(1e6))
    size = [n]

    Z = jax.random.multivariate_normal(key, m, R, size)
    U = jax.scipy.stats.norm.cdf(Z).block_until_ready()

    return jax.device_get(U)


def np_rmvuu(n, rho):
    m = np.zeros(rho.shape[0])

    np.random.seed(np.random.randint(1e6))

    Z = np.random.multivariate_normal(m, rho, n)
    U = scipy.stats.norm.cdf(Z)

    return U


def _u2m(u, margin):
    return margin.ppf(u)


def rvec(n, rho, margins, cores=1, ensure_PSD=False):
    if ensure_PSD:
        rho = cor_nearPSD(rho)
    n = int(n)
    d = rho.shape[0]

    if os.name == 'nt':
        U = np_rmvuu(n, rho)
    else:
        U = jax_rmvuu(n, rho)

    rv = np.empty_like(U)
    if cores <= 1 or os.name == 'nt':
        for i in range(d):
            rv[:, i] = margins[i].ppf(U[:, i])
    else:
        pool = mp.Pool(processes=cores)
        rv = [pool.apply_async(_u2m, args=(U[:, i], margins[i],)) for i in range(d)]
        pool.close()
        rv = [p.get() for p in rv]
        rv = np.stack(rv, axis=1)

    return rv
