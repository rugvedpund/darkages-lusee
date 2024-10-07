# %%
import ipdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from scipy.stats import multivariate_normal
from xarray_einstats import einops, linalg

import simutils

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
scree_vars = ["eva:f-mode", "mean ulsa", "mean da", "mean cmb", "rms ulsa"]


def logpdf(vec_slice, cov_slice):
    mean = np.zeros_like(vec_slice)
    return multivariate_normal.logpdf(vec_slice, mean=mean, cov=cov_slice)


##---------------------------------------------------------------------------##
# %%

# # NOTE: argparser needs something like: python simtensors.py --config configs/a{0..80..10}_dt3600.yaml
# #
# # Load the lusee ulsa sim tensor
# print("Loading ULSA sim tensor..")
# parser = simutils.create_parser()
# args = parser.parse_args()
# sim = simutils.SimTensor().from_args(args)
# print("  saving as netcdf..")
# sim.to_netcdf("netcdf/sim.nc")

# %%

# WARN: when loading netcdf tensor, ensure string coordniates are loaded as strings
#
print("loading netcdf sim tensor..")
sim = xr.open_dataset("netcdf/sim.nc", chunks={"times": 50})
print(sim)
sim["combs"] = sim["combs"].astype(str)
ulsa = sim["ulsa"]
#
# WARN: subtracting mean here, changes how some plots look
# BUG: xarray svd does not rename dims correctly, ensure rename core tensor dims to primed indices
#
print("hosvd..")
print("  subtracting mean..")
tensor = xr.Dataset({"ulsa": ulsa - ulsa.mean("times")})
print("  unfolding..")
tensor["a-mode"] = tensor["ulsa"].stack(tfc=("times", "freqs", "combs"))
tensor["c-mode"] = tensor["ulsa"].stack(tfa=("times", "freqs", "angles"))
tensor["f-mode"] = tensor["ulsa"].stack(tac=("times", "angles", "combs"))
tensor["t-mode"] = tensor["ulsa"].stack(fac=("freqs", "angles", "combs"))
print("  unfolded svd..")
Ua, Sa, _ = np.linalg.svd(tensor["a-mode"], full_matrices=False)
Uc, Sc, _ = np.linalg.svd(tensor["c-mode"], full_matrices=False)
Uf, Sf, _ = np.linalg.svd(tensor["f-mode"], full_matrices=False)
Ut, St, _ = np.linalg.svd(tensor["t-mode"], full_matrices=False)
tensor["eig:freqs"] = xr.DataArray(Uf, dims=["freqs", "freqs2"])
tensor["eig:angles"] = xr.DataArray(Ua, dims=["angles", "angles2"])
tensor["eig:combs"] = xr.DataArray(Uc, dims=["combs", "combs2"])
tensor["eig:times"] = xr.DataArray(Ut, dims=["times", "times2"])
tensor["eva:freqs"] = xr.DataArray(Sf, dims=["freqs2"])
tensor["eva:angles"] = xr.DataArray(Sa, dims=["angles2"])
tensor["eva:combs"] = xr.DataArray(Sc, dims=["combs2"])
tensor["eva:times"] = xr.DataArray(St, dims=["times2"])
print("  calc core and core products..")
tensor["core"] = (
    tensor["eig:freqs"].T
    @ tensor["eig:angles"].T
    @ tensor["eig:combs"].T
    @ (tensor["eig:times"].T @ tensor["ulsa"])
)
tensor["corexUt"] = tensor["core"] @ tensor["eig:times"]
tensor["corexUtxUc"] = tensor["core"] @ tensor["eig:times"] @ tensor["eig:combs"]
tensor["corexUtxUa"] = tensor["core"] @ tensor["eig:times"] @ tensor["eig:angles"]
tensor["corexUtxUf"] = tensor["core"] @ tensor["eig:times"] @ tensor["eig:freqs"]
tensor["corexUtxUaxUc"] = (
    tensor["core"] @ tensor["eig:times"] @ tensor["eig:angles"] @ tensor["eig:combs"]
)
tensor["corexUtxUfxUa"] = (
    tensor["core"] @ tensor["eig:times"] @ tensor["eig:freqs"] @ tensor["eig:angles"]
)
tensor["corexUtxUfxUc"] = (
    tensor["core"] @ tensor["eig:times"] @ tensor["eig:freqs"] @ tensor["eig:combs"]
)
tensor["corexUtxUfxUaxUc"] = (
    tensor["core"]
    @ tensor["eig:times"]
    @ tensor["eig:freqs"]
    @ tensor["eig:angles"]
    @ tensor["eig:combs"]
)
print("calc scree components..")
tensor["ulsa_pfreq"] = tensor["eig:freqs"] @ sim["ulsa"]
tensor["ulsa_pall"] = (
    tensor["eig:freqs"] @ tensor["eig:angles"] @ tensor["eig:combs"] @ sim["ulsa"]
)
tensor["ulsa_nomean_pall"] = (
    tensor["eig:freqs"] @ tensor["eig:angles"] @ tensor["eig:combs"] @ tensor["ulsa"]
)
tensor["ulsa_nomean_pfreq"] = tensor["eig:freqs"] @ tensor["ulsa"]
print("calc covariances..")
tensor["corr_pfreq"] = xr.corr(
    tensor["ulsa_nomean_pfreq"],
    tensor["ulsa_nomean_pfreq"].rename(dict(freqs2="freqs2_")),
    dim=["times"],
)
tensor["corr_pall"] = xr.corr(
    tensor["ulsa_nomean_pall"],
    tensor["ulsa_nomean_pall"].rename(dict(freqs2="freqs2_")),
    dim=["times"],
)
print("saving as netcdf..")
tensor.to_netcdf("netcdf/sim_hosvd.nc")

# %%

# # WARN: when loading netcdf tensor, ensure string coordniates are loaded as strings
# #
# print("loading netcdf hosvd sim tensor..")
# tensor = xr.open_dataset("netcdf/sim_hosvd.nc")
# print("loading sim tensor..")
# sim = xr.open_dataset("netcdf/sim.nc")
# sim["combs"] = sim["combs"].astype(str)
# tensor["combs"] = tensor["combs"].astype(str)
# print("calc scree components..")
# Uf, Ua, Uc = tensor["eig:freqs"], tensor["eig:angles"], tensor["eig:combs"]
# print("  proj freqs only..")
# scree = xr.Dataset(
#     {
#         "eva:f-mode": tensor["eva:freqs"],
#         "mean ulsa": np.abs((Uf @ sim["ulsa"]).mean("times")),
#         "rms ulsa": np.sqrt((Uf @ sim["ulsa"]).var("times")),
#         "mean da": np.abs((Uf @ sim["da"]).mean("times")),
#         "mean cmb": np.abs((Uf @ sim["cmb"]).mean("times")),
#     }
# ).sel(combs=comblist)
# print("  saving as netcdf..")
# scree.to_netcdf("netcdf/sim_scree_pfreq.nc")
# print("  proj all..")
# pscree = xr.Dataset(
#     {
#         "eva:f-mode": tensor["eva:freqs"],
#         "mean ulsa": np.abs((Ua @ Uc @ Uf @ sim["ulsa"]).mean("times")),
#         "rms ulsa": np.sqrt((Ua @ Uc @ Uf @ sim["ulsa"]).var("times")),
#         "mean da": np.abs((Ua @ Uc @ Uf @ sim["da"]).mean("times")),
#         "mean cmb": np.abs((Ua @ Uc @ Uf @ sim["cmb"]).mean("times")),
#     }
# )
# print("  saving as netcdf..")
# pscree.to_netcdf("netcdf/sim_scree_pall.nc")

# %%

print("loading netcdf tensors..")
sim = xr.open_dataset("netcdf/sim.nc")
sim["combs"] = sim["combs"].astype(str)
# tensor = xr.open_dataset("netcdf/sim_hosvd.nc", chunks={"times": 50})
tensor = xr.open_dataset("netcdf/sim_hosvd.nc")
tensor["combs"] = tensor["combs"].astype(str)
tensor["freqs2"] = np.arange(50)
tensor.load()
scree = xr.open_dataset("netcdf/sim_scree_pall.nc")
scree["freqs2"] = np.arange(50)
# %%

tensor["ulsa_nomean_pall"].sel(combs2=0, angles2=0).plot.pcolormesh(
    norm=symlog, cmap="viridis"
)
plt.show()

# %%

tensor["cov_pall"].sel(combs2=[0], angles2=[0]).plot.imshow(
    norm=symlog,
    col="combs2",
    row="angles2",
)
plt.show()

# %%

# # f2s = np.arange(50)
# f2s = np.arange(0, 50, 10)
# c2, a2 = 15, 8
# d = np.vstack(
#     [
#         tensor["ulsa_nomean_pall"].sel(freqs2=f2s, combs2=c2, angles2=a2).data.T,
#         scree["mean ulsa"].sel(freqs2=f2s, combs2=c2, angles2=a2).data,
#         scree["mean da"].sel(freqs2=f2s, combs2=c2, angles2=a2).data,
#         scree["mean cmb"].sel(freqs2=f2s, combs2=c2, angles2=a2).data,
#     ]
# )
# indx = ["data"] * 650 + ["mean ulsa", "mean da", "mean cmb"]
# df = pd.DataFrame(d, index=indx).reset_index()
# kwargs = {
#     "markers": [".", "d", "^", "v"],
#     "height": 2,
#     # "vars": "times",
#     "hue": "index",
#     "diag_kind": "hist",
# }
# pairplt = sns.pairplot(df, **kwargs)
# plt.show()
# # plt.savefig("tex/figures/pairplot_pa0_pc0.png")

# %%


# %%

##---------------------------------------------------------------------------##
# calculate tensors and save
##---------------------------------------------------------------------------##

# scree = xr.open_dataset("netcdf/sim_scree_pfreq.nc")
# scree["combs"] = scree["combs"].astype(str)
# pscree = xr.open_dataset("netcdf/sim_scree_pall.nc")

##---------------------------------------------------------------------------##
# Plots
##---------------------------------------------------------------------------##
# %%

# # plot frequency covariances
# tensor["corr_pfreq"].sel(combs=comblist).plot.imshow(
#     col="combs", row="angles", norm=symlog, size=2
# )
# plt.savefig("tex/figures/cov_pfreq.pdf")
# plt.show()
#
# %%

# tensor["corr_pall"].plot.imshow(
#     col="combs2", row='angles2', norm=symlog, size=2
# )
# plt.savefig("tex/figures/cov_pall.pdf")
# plt.show()
#
# %%
# pscree.to_array().sel(variable=scree_vars).plot.line(
#     row="angles2", col="combs2", hue="variable", yscale="log", size=4, aspect=0.5
# )
# plt.savefig("tex/figures/simscree_pall.pdf")

# scree.to_array().sel(variable=scree_vars).plot.line(
#     row="angles", col="combs", hue="variable", yscale="log", size=4, aspect=0.5
# )
# plt.savefig("tex/figures/simscree_pfreq.pdf")

# # plot eig:freqs
# tensor["eig:freqs"].sel(freqs2=[0, 1, 2, 3, 4]).plot.line( hue="freqs2", aspect=3.0, size=6)
# plt.gca().get_legend().set_title(None)
# plt.grid()
# plt.show()
# # plt.savefig("tex/figures/simeigfreqs.pdf")

# # plot eig:combs
# tensor["eig:combs"].sel(combs=comblist, combs2=[0, 1, 2, 3, 4]).plot.line( hue="combs2", aspect=2.0, size=6)
# plt.gca().get_legend().set_title(None)
# plt.grid()
# plt.show()
# # plt.savefig("tex/figures/simeigcombs.pdf")

# # plot eig:angles
# tensor["eig:angles"].sel(angles2=[0, 1, 2, 3, 4]).plot.line(hue="angles2")
# plt.gca().get_legend().set_title(None)
# plt.show()
# # plt.savefig('tex/figures/simeigangles.pdf')

# # plot everything
# tensor['corexUtxUfxUaxUc'].sel(combs=comblist).plot.imshow(row='angles',col='combs',norm=symlog,aspect=0.5,y='freqs',cmap='viridis')
# plt.show()
# # plt.savefig('tex/figures/simall.pdf')

# # plot all combs, single angle
# ulsa.sel(angles=0,combs=combmat).plot.imshow(col='combs',col_wrap=4,norm=symlog,cmap='viridis',y='freqs',robust=True,aspect=2.0,size=2)
# plt.show()
# # plt.savefig('tex/figures/sim0.pdf')

# # # plot all angles, single comb
# ulsa.sel(combs="00R").plot.imshow(
#     col="angles", col_wrap=3, y="freqs", aspect=2.0, norm=symlog, cmap="viridis", aspect=2.0, size=2
# )
# plt.show
# # plt.savefig('tex/figures/simC00R.pdf')
