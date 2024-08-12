# %%

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import multivariate_normal
from xarray_einstats import einops, linalg

import simutils

##--------------------------------------------------------------------##
# sim stores the original lusee sim data
# eigs stores the projection to orthogonal combs and angles
# psim stores those projections
##--------------------------------------------------------------------##
print("loading netcdf tensors..")
sim = xr.open_dataset("netcdf/sim.nc", chunks={"times": 650})
sim["combs"] = sim["combs"].astype(str)
sim = sim.sel(combs=simutils.comblist)
with open("netcdf/sim_hosvd.nc", "rb") as f:
    eigs = xr.open_dataset(f)[["eig:combs", "eig:angles"]]
psim = xr.Dataset()
psim["ulsa"] = sim["ulsa"] @ eigs["eig:combs"] @ eigs["eig:angles"]
psim["da"] = sim["da"] @ eigs["eig:combs"] @ eigs["eig:angles"]
psim["cmb"] = sim["cmb"] @ eigs["eig:combs"] @ eigs["eig:angles"]

data = psim["ulsa"] - psim["ulsa"].mean("times")

# tensor = xr.open_dataarray("netcdf/sim_projcombsangles.nc", chunks={"times": 650})
# tensor.load()
# # this only existed as mean subtracted sim tensor

# %%


##--------------------------------------------------------------------##
# get pca using times_dim as individual data points, freqs_dim as features
# specify other_dims to keep/ignore
# to be used for the individual projections
##--------------------------------------------------------------------##
def get_pca(
    datatensor: xr.DataArray,
    times_dim: str,
    freqs_dim: str,
    other_dims: [str],
) -> xr.Dataset:
    out = xr.Dataset()
    assert times_dim in datatensor.dims
    assert freqs_dim in datatensor.dims
    assert all(dim in datatensor.dims for dim in other_dims)
    # NOTE: mean subtracted tensor
    out["delta"] = datatensor - datatensor.mean(times_dim)
    out["delta"].load()
    print("doing pca..")
    out["Uf"], out["Sf"], _ = out["delta"].linalg.svd(
        dims=(freqs_dim, times_dim),
        full_matrices=False,
        out_append="_eig",
    )
    out["Sf"] = out["Sf"].rename({freqs_dim: f"{freqs_dim}_eig"})
    out["delta_eig"] = xr.dot(out["Uf"], out["delta"], dims=freqs_dim)
    out["eva"] = out["Sf"] / np.sqrt(datatensor.sizes[freqs_dim] - 1)
    return out
    # FIX: really only need eva, eve and delta_eig?


pca = get_pca(
    data, times_dim="times", freqs_dim="freqs", other_dims=["combs2", "angles2"]
)

# %%


def get_pca_proj(
    ulsa: xr.DataArray,
    da: xr.DataArray,
    cmb: xr.DataArray,
    pca_tensor: xr.Dataset,
    times_dim: str,
    freqs_dim: str,
    other_dims: [str],
) -> xr.Dataset:
    print("projecting pca..")
    assert times_dim in ulsa.dims
    assert freqs_dim in ulsa.dims
    assert all(dim in ulsa.dims for dim in other_dims)
    out = xr.Dataset()
    out["pulsa"] = xr.dot(ulsa, pca_tensor["Uf"], dims=freqs_dim)
    out["pda"] = xr.dot(da, pca_tensor["Uf"], dims=freqs_dim)
    out["pcmb"] = xr.dot(cmb, pca_tensor["Uf"], dims=freqs_dim)
    out["mean pulsa"] = out["pulsa"].mean(times_dim)
    out["mean pda"] = out["pda"].mean(times_dim)
    out["mean pcmb"] = out["pcmb"].mean(times_dim)
    out["rms pulsa"] = out["pulsa"].std(times_dim)
    return out


proj = get_pca_proj(
    psim["ulsa"],
    psim["da"],
    psim["cmb"],
    pca,
    times_dim="times",
    freqs_dim="freqs",
    other_dims=["combs2", "angles2"],
)

# %%


def get_gaussian_cov(
    proj_tensor: xr.Dataset,
    times_dim: str,
    freqs_dim: str,
    other_dims: [str],
) -> xr.DataArray:
    assert times_dim in proj_tensor["pulsa"].dims
    assert freqs_dim in proj_tensor["pulsa"].dims
    assert all(dim in proj_tensor["pulsa"].dims for dim in other_dims)
    print("getting gauss cov..")
    norm_pulsa = proj_tensor["pulsa"] / proj_tensor["rms pulsa"]
    out = xr.cov(
        norm_pulsa,
        norm_pulsa.rename(
            {freqs_dim: f"{freqs_dim}(copy)"}
        ),  # FIX: make a function in utils
        dim=[times_dim],
    )
    return out


gausscov = get_gaussian_cov(
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


def get_gaussianapprox_likelihood(
    proj_tensor: xr.Dataset,
    gauss_cov: xr.DataArray,
    amp_dimless: xr.DataArray,
    freqs_dim: str,
    other_dims: [str],
    which: str = "both",
) -> xr.Dataset:
    assert freqs_dim in proj_tensor.dims
    assert all(dim in proj_tensor.dims for dim in other_dims)
    print("getting gaussianapprox likelihood..")
    loglike = xr.Dataset()
    like = xr.Dataset()
    print("  da..")
    if which == "both" or which == "da":
        loglike["da"] = xr.apply_ufunc(
            multivariate_normal.logpdf,
            proj_tensor["mean pulsa"] + (1 - amp_dimless) * proj_tensor["mean pda"],
            xr.DataArray(np.zeros(50), dims=freqs_dim),
            gauss_cov,
            input_core_dims=[
                [freqs_dim],
                [freqs_dim],
                [freqs_dim, f"{freqs_dim}(copy)"],
            ],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
        )
        like["da"] = np.exp(loglike["da"] - loglike["da"].max("amp"))
    print("  cmb..")
    if which == "both" or which == "cmb":
        loglike["cmb"] = xr.apply_ufunc(
            multivariate_normal.logpdf,
            proj_tensor["mean pulsa"] + (1 - amp_dimless) * proj_tensor["mean pcmb"],
            xr.DataArray(np.zeros(50), dims=freqs_dim),
            gauss_cov,
            input_core_dims=[
                [freqs_dim],
                [freqs_dim],
                [freqs_dim, f"{freqs_dim}(copy)"],
            ],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
        )
        like["cmb"] = np.exp(loglike["cmb"] - loglike["cmb"].max("amp"))
    like.coords["amp"] = amp_dimless

    return like


# %%


# FIX: feels like a hack
def np_symlogspace(start, stop, num=50):
    return np.sort(
        np.concatenate(
            [
                -np.logspace(start, stop, num=num // 2),
                np.logspace(start, stop, num=num // 2),
            ]
        )
    )


# %%

# amp = xr.DataArray(np_symlogspace(-10, 10, num=1000), dims="amp")
amp = xr.DataArray(np.linspace(-1e9, 1e9, num=1000), dims="amp")
like = get_gaussianapprox_likelihood(
    proj,
    gausscov,
    amp,
    freqs_dim="freqs_eig",
    other_dims=["combs2", "angles2"],
    which="da",
)
like.load()

like["da"].plot.line(x="amp", col="combs2", hue="angles2", col_wrap=4)
plt.savefig("tex/figures/da_angles.pdf")
# plt.show()
like["da"].plot.line(x="amp", col="angles2", hue="combs2", col_wrap=3)
plt.savefig("tex/figures/da_combs.pdf")
plt.show()

# %%

# amp = xr.DataArray(np_symlogspace(-10, 10, num=1000), dims="amp")
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

like["da"] = np.exp(loglike["da"] - loglike["da"].max("amp"))
like["cmb"].plot.line(x="amp", col="combs2", hue="angles2", col_wrap=4)
plt.savefig("tex/figures/cmb_angles.pdf")
# plt.show()
like["cmb"].plot.line(x="amp", col="angles2", hue="combs2", col_wrap=3)
plt.savefig("tex/figures/cmb_combs.pdf")
plt.show()


# %%
