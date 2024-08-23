# %%

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import multivariate_normal
from xarray_einstats import einops, linalg

import simflows.pca as simpca
import simflows.utils as simutils

##--------------------------------------------------------------------##
# sim stores the original lusee sim data
# eigs stores the projection to orthogonal combs and angles
# psim stores those projections
##--------------------------------------------------------------------##

print("loading netcdf tensors..")
sim = xr.open_dataset("data/netcdf/sim.nc", chunks={"times": 650})
sim["combs"] = sim["combs"].astype(str)
sim = sim.sel(combs=simutils.comblist)
with open("data/netcdf/sim_hosvd.nc", "rb") as f:
    eigs = xr.open_dataset(f)[["eig:combs", "eig:angles"]]
psim = xr.Dataset()
psim["ulsa"] = sim["ulsa"] @ eigs["eig:combs"] @ eigs["eig:angles"]
psim["da"] = sim["da"] @ eigs["eig:combs"] @ eigs["eig:angles"]
psim["cmb"] = sim["cmb"] @ eigs["eig:combs"] @ eigs["eig:angles"]
data = psim["ulsa"] - psim["ulsa"].mean("times")

# %%

pca = simpca.get_pca(
    data, times_dim="times", freqs_dim="freqs", other_dims=["combs2", "angles2"]
)
pca

# %%

proj = simpca.get_pca_proj(
    psim["ulsa"],
    psim["da"],
    psim["cmb"],
    pca,
    times_dim="times",
    freqs_dim="freqs",
    other_dims=["combs2", "angles2"],
)
proj

# %%


gausscov = simpca.get_gaussian_cov(
    proj,
    times_dim="times",
    freqs_dim="freqs_eig",
    other_dims=["combs2", "angles2"],
)
gausscov

# %%

gausscov.plot.imshow(x="freqs_eig", y="freqs_eig(copy)", col="combs2", row="angles2")
plt.show()

# %%

# %%
##--------------------------------------------------------------------##
# dark ages

# amp = xr.DataArray(simutils.np_symlogspace(-10, 10, num=1000), dims="amp")
amp = xr.DataArray(np.linspace(-1e9, 1e9, num=1000), dims="amp")
like = simpca.get_gaussianapprox_likelihood(
    proj,
    gausscov,
    amp,
    freqs_dim="freqs_eig",
    other_dims=["combs2", "angles2"],
    which="da",
)
like.load()

# %%

# like["da"].plot.line(x="amp", col="combs2", hue="angles2", col_wrap=4)
# # plt.savefig("olddocs/tex/figures/da_angles.pdf")
# plt.show()
# like["da"].plot.line(x="amp", col="angles2", hue="combs2", col_wrap=3)
# # plt.savefig("olddocs/tex/figures/da_combs.pdf")
# plt.show()

like["da"].sum(["angles2", "combs2"]).plot.line(x="amp")
plt.savefig("olddocs/tex/figures/da_sum.pdf")
plt.show()
##--------------------------------------------------------------------##


# %%
##--------------------------------------------------------------------##
# cmb

# amp = xr.DataArray(simutils.np_symlogspace(-10, 10, num=1000), dims="amp")
amp = xr.DataArray(np.linspace(-1e7, 1e7, num=1000), dims="amp")
like = get_gaussianapprox_likelihood(
    proj,
    gausscov,
    amp,
    freqs_dim="freqs_eig",
    other_dims=["combs2", "angles2"],
    which="cmb",
)
like.load()

# %%

# like["cmb"].plot.line(x="amp", col="combs2", hue="angles2", col_wrap=4)
# plt.savefig("olddocs/tex/figures/cmb_angles.pdf")
# # plt.show()
# like["cmb"].plot.line(x="amp", col="angles2", hue="combs2", col_wrap=3)
# plt.savefig("olddocs/tex/figures/cmb_combs.pdf")
# # plt.show()

like["cmb"].product(["angles2", "combs2"]).plot.line(x="amp")
plt.savefig("olddocs/tex/figures/cmb_sum.pdf")
plt.show()
##--------------------------------------------------------------------##


# %%

amp = xr.DataArray(simutils.np_symlogspace(-10, 10, num=1000), dims="amp")
# amp = xr.DataArray(np.linspace(-1e7, 1e7, num=1000), dims="amp")
likeboth = simpca.get_gaussianapprox_likelihood(
    proj,
    gausscov,
    amp,
    freqs_dim="freqs_eig",
    other_dims=["combs2", "angles2"],
    which="both",
)
likeboth.load()

# %%

# %%

# %%
