import jax
import numpy as np
from bigsimr.cor_utils import cor_convert
from bigsimr.cor_nearPSD import cor_nearPSD


def _rmvuu(n, rho):
    R = jax.device_put(np.array(rho))
    m = jax.numpy.zeros(rho.shape[1])

    key = jax.random.PRNGKey(np.random.randint(1e6))
    size = [n]

    Z = jax.random.multivariate_normal(key, m, R, size)
    U = jax.scipy.stats.norm.cdf(Z).block_until_ready()

    return jax.device_get(U)


def rvec(n, rho, margins, type="pearson", ensure_PSD=False):
    types = ['pearson', 'spearman', 'kendall']
    if type not in types:
        raise ValueError("Invalid type. Expected one of: %s" % types)

    if type == "spearman" or type == "kendall":
        rho = cor_convert(rho, type + '_' + 'pearson')

    if ensure_PSD:
        rho = nearPD(rho)

    U = _rmvuu(n, rho)
    d = rho.shape[0]

    rv = np.zeros_like(U)
    for i in range(d):
        rv[:, i] = margins[i].ppf(U[:, i])

    return rv
