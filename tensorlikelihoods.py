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

# simtensor["S_t_f_a2_c2"] = tensor["ulsa"] @ tensor["eig:combs"] @ tensor["eig:angles"]
# simtensor["eve_f_f2_a2_c2"], simtensor["eva_a2_c2_f2"], _ = simtensor[
#     "S_t_f_a2_c2"
# ].linalg.svd(dims=("freqs", "times"), full_matrices=False, out_append="_eig")
# simtensor["eva_a2_c2_f2"] = simtensor["eva_a2_c2_f2"].rename(freqs="freqs_eig")
# simtensor["S_t_f2_a2_c2"] = xr.dot(
#     simtensor["eve_f_f2_a2_c2"], simtensor["S_t_f_a2_c2"], dims="freqs"
# )
# simtensor["pulsa"] = xr.dot(
#     sim["ulsa"]
#     @ tensor["eig:combs"]
#     @ tensor["eig:angles"],  # w/mean proj to comb, angle eigspace
#     simtensor["eve_f_f2_a2_c2"],  # proj to freq eigspace per angle, comb eigspace
#     dims="freqs",
# )
# simtensor["pda"] = xr.dot(
#     sim["da"] @ tensor["eig:combs"] @ tensor["eig:angles"],
#     simtensor["eve_f_f2_a2_c2"],
#     dims="freqs",
# )
# simtensor["pcmb"] = xr.dot(
#     sim["cmb"] @ tensor["eig:combs"] @ tensor["eig:angles"],
#     simtensor["eve_f_f2_a2_c2"],
#     dims="freqs",
# )
# simtensor["mean ulsa"] = simtensor["pulsa"].mean("times")
# simtensor["mean da"] = simtensor["pda"].mean("times")
# simtensor["mean cmb"] = simtensor["pcmb"].mean("times")
# simtensor["rms ulsa"] = np.sqrt(simtensor["pulsa"].var("times"))
# simtensor["eva"] = (simtensor["eva_a2_c2_f2"]) / np.sqrt(650 - 1)

# %%

# np.abs(simtensor[scree_vars]).to_array().plot.line(
#     x="freqs_eig",
#     col="combs2",
#     row="angles2",
#     hue="variable",
#     yscale="log",
#     aspect=0.5,
# )
# # plt.savefig("tex/figures/simscree_panglescombs.pdf")
# plt.show()

# %%

# import tensorly as tl
# wts, facs = tl.decomposition.parafac(
#     (simtensor["mean ulsa"]+simtensor['mean da']+simtensor['mean cmb']).data, rank=3, normalize_factors=True
# )
# print(wts)
# for fac in facs:
#     plt.plot(fac, "o-")
#     plt.show()

# %%


def get_pca(
    tensor: xr.DataArray,
    times_dim: str = "times",
    freqs_dim: str = "freqs",
    other_dims: [str] = ["combs", "angles"],
) -> xr.Dataset:
    out = xr.Dataset()
    assert times_dim in tensor.dims
    assert freqs_dim in tensor.dims
    assert all(dim in tensor.dims for dim in other_dims)
    out["delta"] = tensor - tensor.mean(times_dim)
    out["Uf"], out["Sf"], _ = out["delta"].linalg.svd(
        dims=(freqs_dim, times_dim), full_matrices=False, out_append="_eig"
    )
    out["Sf"] = out["Sf"].rename({freqs_dim: f"{freqs_dim}_eig"})
    out["delta_eig"] = xr.dot(out["Uf"], out["delta"], dims=freqs_dim)
    out["eva"] = out["Sf"] / np.sqrt(tensor.sizes[freqs_dim] - 1)
    return out


def get_pca_proj(
    ulsa: xr.DataArray, da: xr.DataArray, cmb: xr.DataArray, pca_tensor: xr.Dataset
) -> xr.Dataset:
    out = xr.Dataset()
    out["pulsa"] = xr.dot(ulsa, pca_tensor["Uf"], dims="freqs")
    out["pda"] = xr.dot(da, pca_tensor["Uf"], dims="freqs")
    out["pcmb"] = xr.dot(cmb, pca_tensor["Uf"], dims="freqs")
    out["mean ulsa"] = out["pulsa"].mean("times")
    out["mean da"] = out["pda"].mean("times")
    out["mean cmb"] = out["pcmb"].mean("times")
    out["rms ulsa"] = out["pulsa"].std("times")
    return out


def get_gaussianapprox_likelihood(
    proj_tensor: xr.Dataset,
    amp_array: xr.DataArray,
    times_dim: str = "times",
    freqs_dim: str = "freqs_eig",
    other_dims: [str] = ["combs", "angles"],
) -> xr.Dataset:
    out = xr.Dataset()
    out["norm_pulsa"] = proj_tensor["pulsa"] / proj_tensor["rms ulsa"]
    print("getting cov..")
    out["cov"] = xr.cov(
        out["norm_pulsa"],
        out["norm_pulsa"].rename({freqs_dim: f"{freqs_dim}_2"}),
        dim=[times_dim],
    )

    return out


# get_pca(sim["ulsa"])
proj = get_pca_proj(sim["ulsa"], sim["da"], sim["cmb"], get_pca(sim["ulsa"]))
amp = xr.DataArray(np.linspace(-1e3, 1e3, num=1000), dims="amp")
gauss = get_gaussianapprox_likelihood(proj, amp)
gauss

# %%

multivariate_normal.pdf(
    np.ones(50), mean=np.zeros(50), cov=gauss["cov"].sel(combs="11R", angles=0).data
)

# %%

xr.apply_ufunc(
    multivariate_normal.logpdf,
    np.ones(50),
    np.zeros(50),
    gauss["cov"],
    input_core_dims=[[], [], ["freqs_eig", "freqs_eig_2"]],
    output_core_dims=[[]],
)

# %%
