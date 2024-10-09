# %%

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import multivariate_normal
from xarray_einstats import einops, linalg

import simflows as sf

# %%

log = mpl.colors.LogNorm()
symlog = mpl.colors.SymLogNorm(linthresh=1e2)
combmat = [
    "00R",
    "01R",
    "02R",
    "03R",
    "01I",
    "11R",
    "12R",
    "13R",
    "02I",
    "12I",
    "22R",
    "23R",
    "03I",
    "13I",
    "23I",
    "33R",
]
combauto = ["00R", "11R", "22R", "33R"]
comblist = [
    "00R",
    "11R",
    "22R",
    "33R",
    "01R",
    "02R",
    "03R",
    "12R",
    "13R",
    "23R",
    "01I",
    "02I",
    "03I",
    "12I",
    "13I",
    "23I",
]
scree_vars = ["rms ulsa", "mean ulsa", "mean da", "mean cmb"]

# %%

print("loading netcdf tensors..")
sim = xr.open_dataset("netcdf/sim.nc")
sim["combs"] = sim["combs"].astype(str)
sim = sim.sel(combs=comblist)
tensor = xr.open_dataset("netcdf/sim_hosvd.nc")
tensor["combs"] = tensor["combs"].astype(str)
tensor = tensor.sel(combs=comblist)
tensor["freqs2"] = np.arange(50)
tensor.load()

simtensor = xr.Dataset()
simtensor["mean ulsa"] = sim["ulsa"].mean("times")
simtensor["mean da"] = sim["da"].mean("times")
simtensor["mean cmb"] = sim["cmb"].mean("times")
simtensor["rms ulsa"] = sim["ulsa"].std("times")
simtensor["delta ulsa"] = sim["ulsa"] - sim["ulsa"].mean("times")

# %%


# %%

import os

for file in os.listdir("netcdf"):
    print(f"\n\n{file}")
    xrtensor = xr.open_dataset(f"netcdf/{file}")
    print(xrtensor)


# %%
