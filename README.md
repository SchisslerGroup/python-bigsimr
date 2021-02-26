# bigsimr

`bigsimr` is a Python3 package for simulating high-dimensional multivariate data with a target correlation and arbitrary marginal distributions via Gaussian copula. It utilizes [Bigsimr.jl](https://github.com/adknudson/Bigsimr.jl) for its core routines. For full documentation and examples, please see the [Bigsimr.jl docs](https://adknudson.github.io/Bigsimr.jl/stable/).

## Features

* **Pearson matching** - employs a matching algorithm (Xiao and Zhou 2019) to account for the non-linear transformation in the Normal-to-Anything (NORTA) step
* **Spearman and Kendall matching** - Use explicit transformations (Lebrun and Dutfoy 2009)
* **Nearest Correlation Matrix** - Calculate the nearest positive [semi]definite correlation matrix (Qi and Sun 2006)
* **Fast Approximate Correlation Matrix** - Calculate an approximation to the nearest positive definite correlation matrix
* **Random Correlation Matrix** - Generate random positive [semi]definite correlation matrices
* **Fast Multivariate Normal Generation** - Utilize multithreading to generate multivariate normal samples in parallel

## Installation and Setup

Install the `bigsimr` package from pip using

```
pip install git+https://github.com/SchisslerGroup/python-bigsimr.git
```

Or install the development version with

```
pip install git+https://github.com/SchisslerGroup/python-bigsimr.git@dev
```

`bigsimr` relies on the Julia language to execute code through the python `julia` package. Julia can be obtained from [julialang.org](https://julialang.org/downloads/), or it can be detected/installed automatically using the setup function provided by `bigsimr`. The `setup()` function will also install the required Julia packages for bigsimr.

```python
from bigsimr import setup
setup(compiled_modules=False)
```

**Note.** The `compiled_modules=False` argument is necessary for those using Python from a conda environment. There is a known bug where setup fails if `compiled_modules` is set to `True` (the default for the `julia` package).

## Using

```python
from julia.api import Julia
jl = Julia(compiled_modules=False) # conda users -> set to False

from julia import Bigsimr as bs
from julia import Distributions as dist

import numpy as np
```

### Examples

Pearson mathcing

```python
target_corr = bs.cor_randPD(3)
margins = [dist.Binomial(20, 0.2), dist.Beta(2, 3), dist.LogNormal(3, 1)]

adjusted_corr = bs.pearson_match(target_corr, margins)

x = bs.rvec(100_000, adjusted_corr, margins)
bs.cor(x, bs.Pearson)
```

Spearman/Kendall matching

```python
spearman_corr = bs.cor_randPD(3)
adjusted_corr = bs.cor_convert(spearman_corr, bs.Spearman, bs.Pearson)

x = bs.rvec(100_000, adjusted_corr, margins)
bs.cor(x, bs.Spearman)
```

Nearest correlation matrix

```python
from julia.LinearAlgebra import isposdef

s = bs.cor_randPSD(200)
r = bs.cor_convert(s, bs.Spearman, bs.Pearson)
isposdef(r)

p = bs.cor_nearPD(r)
isposdef(p)
```

Fast approximate nearest correlation matrix

```python
s = bs.cor_randPSD(2000)
r = bs.cor_convert(s, bs.Spearman, bs.Pearson)
isposdef(r)

p = bs.cor_fastPD(r)
isposdef(p)
```

# References

* Xiao, Q., & Zhou, S. (2019). Matching a correlation coefficient by a Gaussian copula. Communications in Statistics-Theory and Methods, 48(7), 1728-1747.
* Lebrun, R., & Dutfoy, A. (2009). An innovating analysis of the Nataf transformation from the copula viewpoint. Probabilistic Engineering Mechanics, 24(3), 312-320.
* Qi, H., & Sun, D. (2006). A quadratically convergent Newton method for computing the nearest correlation matrix. SIAM journal on matrix analysis and applications, 28(2), 360-385.
* amoeba (https://stats.stackexchange.com/users/28666/amoeba), How to generate a large full-rank random correlation matrix with some strong correlations present?, URL (version: 2017-04-13): https://stats.stackexchange.com/q/125020