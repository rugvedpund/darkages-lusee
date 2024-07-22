# %%

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import simflows as sf

# %%

print("loading netcdf tensors..")
sim = xr.open_dataset("netcdf/sim.nc")
sim["combs"] = sim["combs"].astype(str)
# tensor = xr.open_dataset("netcdf/sim_hosvd.nc", chunks={"times": 50})
tensor = xr.open_dataset("netcdf/sim_hosvd.nc")
tensor["combs"] = tensor["combs"].astype(str)
tensor["freqs2"] = np.arange(50)
scree = xr.open_dataset("netcdf/sim_scree_pall.nc")
scree["freqs2"] = np.arange(50)

tensor.load()

# %%

cov = xr.cov(tensor["f-mode"], tensor["f-mode"].rename(freqs2="freqs2_"), dim="tac")
print(cov)

# %%
