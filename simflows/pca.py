# Description: PCA functions for the simflows package

import numpy as np
import xarray as xr

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
    if other_dims is not None:
        assert all(dim in datatensor.dims for dim in other_dims)
    # NOTE: mean subtracted tensor
    # out["delta"] = datatensor - datatensor.mean(times_dim)
    # out["delta"].load()

    print("doing pca..")
    U, S, Vt = datatensor.linalg.svd(
        dims=(freqs_dim, times_dim),
        full_matrices=False,
        out_append="_eig",
    )
    S = S.rename({freqs_dim: f"{freqs_dim}_eig"})
    Vt = Vt.rename({freqs_dim: f"{freqs_dim}_eig"})
    return xr.Dataset({"U": U, "S": S, "Vt": Vt})
    # FIX: really only need eva, eve and delta_eig?


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
