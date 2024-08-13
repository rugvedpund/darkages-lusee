import fitsio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import healpy as hp
import simutils
import xarray as xr
from xarray_einstats import linalg
log = mpl.colors.LogNorm()
symlog = mpl.colors.SymLogNorm(linthresh=1e-9)

# Load the ULSA map
print("Loading ULSA map..")
ulsa = fitsio.read("ulsa.fits")
dulsa = np.vstack([hp.ud_grade(ulsa[i,:], nside_out=32) for i in range(50)])
print('getting dumb map-inverse..')
mapinv = np.linalg.pinv(dulsa.T)
print("mapinverse shape:", mapinv.shape)

# Load the lusee ulsa sim tensor
print("Loading ULSA sim tensor..")
parser = simutils.create_parser()
args = parser.parse_args()
sim = simutils.SimTensor().from_args(args)

# Dumb mapmaker
print("Getting dumb mapmaker..")

"""
create sim2map xarray object
multiply each angle, comb with mapinv
find the pinv of the result, for each angle, comb
"""
mapinv = xr.DataArray(mapinv, dims=["freqs2", "pixels"], coords=[sim.tensor.freqs, np.arange(hp.nside2npix(32))])


import ipdb; ipdb.set_trace()


map2sim = sim.tensor.ulsa.sel(angles=0,combs="00R").data @ mapinv
sim2map = np.linalg.pinv(map2sim.T)


