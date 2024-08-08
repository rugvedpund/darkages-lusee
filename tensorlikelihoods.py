# %%

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import multivariate_normal
from xarray_einstats import einops, linalg

import simutils

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
tensor = xr.open_dataarray("netcdf/sim_projcombsangles.nc", chunks={"times": 650})
tensor.load()

# %%


def get_pca(
    tensor: xr.DataArray,
    times_dim: str,
    freqs_dim: str,
    other_dims: [str],
) -> xr.Dataset:
    out = xr.Dataset()
    assert times_dim in tensor.dims
    assert freqs_dim in tensor.dims
    assert all(dim in tensor.dims for dim in other_dims)
    out["delta"] = tensor - tensor.mean(times_dim)
    print("doing pca..")
    out["Uf"], out["Sf"], _ = out["delta"].linalg.svd(
        dims=(freqs_dim, times_dim), full_matrices=False, out_append="_eig"
    )
    out["Sf"] = out["Sf"].rename({freqs_dim: f"{freqs_dim}_eig"})
    out["delta_eig"] = xr.dot(out["Uf"], out["delta"], dims=freqs_dim)
    out["eva"] = out["Sf"] / np.sqrt(tensor.sizes[freqs_dim] - 1)
    return out


def get_pca_proj(
    ulsa: xr.DataArray,
    da: xr.DataArray,
    cmb: xr.DataArray,
    pca_tensor: xr.Dataset,
    times_dim: str,
    freqs_dim: str,
    other_dims: [str],
) -> xr.Dataset:
    # FIX: assert freqs and times dims are appropriate otherwise long computation
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
        norm_pulsa.rename({freqs_dim: f"{freqs_dim}(copy)"}),
        dim=[times_dim],
    )
    return out


# %%

pca = get_pca(
    tensor, times_dim="times", freqs_dim="freqs", other_dims=["combs2", "angles2"]
)
proj = get_pca_proj(
    psim["ulsa"],
    psim["da"],
    psim["cmb"],
    pca,
    times_dim="times",
    freqs_dim="freqs",
    other_dims=["combs2", "angles2"],
)
gauss = get_gaussian_cov(
    proj,
    times_dim="times",
    freqs_dim="freqs_eig",
    other_dims=["combs2", "angles2"],
)
gauss


# %%


def get_gaussianapprox_likelihood(
    proj_tensor: xr.Dataset,
    gauss_cov: xr.DataArray,
    amp: xr.DataArray,
    freqs_dim: str,
    other_dims: [str],
) -> xr.Dataset:
    assert freqs_dim in proj_tensor["pulsa"].dims
    assert all(dim in proj_tensor["pulsa"].dims for dim in other_dims)
    print("getting gaussianapprox likelihood..")


# %%

amp = xr.DataArray(np.linspace(-1e3, 1e3, num=1000), dims="amp")
xr.apply_ufunc(
    multivariate_normal.logpdf,
    proj["mean pulsa"],
    xr.DataArray(np.zeros(50), dims="freqs_eig"),
    gauss["cov"],
    input_core_dims=[["freqs_eig"], ["freqs_eig"], ["freqs_eig", "freqs_eig(copy)"]],
    output_core_dims=[[]],
    vectorize=True,
    dask="parallelized",
)


# %%

multivariate_normal.pdf(
    np.ones(50), mean=np.zeros(50), cov=gauss["cov"].sel(combs2=0, angles2=0).data
)
