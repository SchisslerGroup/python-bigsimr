import julia
from julia.api import Julia


def setup():
    julia.install()

    jl = Julia(compiled_modules=False)

    from julia import Pkg
    Pkg.add("Distributions")
    Pkg.add("Bigsimr")
