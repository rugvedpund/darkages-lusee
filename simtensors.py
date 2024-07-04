import lusee
import simutils
import simflows as sf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
from xarray_einstats import linalg, einops
import tensorly as tl

##---------------------------------------------------------------------------##

if __name__ == "__main__":
    log = mpl.colors.LogNorm()
    symlog = mpl.colors.SymLogNorm(linthresh=1e2)

    parser = simutils.create_parser()
    args = parser.parse_args()

    sim = simutils.SimTensor().from_args(args)

    import ipdb; ipdb.set_trace()
