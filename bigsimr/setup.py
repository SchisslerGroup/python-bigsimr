import julia
from julia.api import Julia


def setup(compiled_modules=True):
    julia.install()
    jl = Julia(compiled_modules=compiled_modules)

    from julia import Pkg
    Pkg.add("Bigsimr@0.8.0") # Lock to specific version for stability
    Pkg.add("Distributions") # Install Distributions after Bigsimr
