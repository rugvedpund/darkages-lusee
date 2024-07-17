import ipdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xarray_einstats import einops, linalg

import simutils

log = mpl.colors.LogNorm()
symlog = mpl.colors.SymLogNorm(linthresh=1e2)

##---------------------------------------------------------------------------##

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


print("loading netcdf tensors..")
sim = xr.open_dataset("netcdf/sim.nc")
sim["combs"] = sim["combs"].astype(str)
tensor = xr.open_dataset("netcdf/sim_hosvd.nc")
tensor["combs"] = tensor["combs"].astype(str)
scree = xr.open_dataset("netcdf/sim_scree_pfreq.nc")
scree["combs"] = scree["combs"].astype(str)
pscree = xr.open_dataset("netcdf/sim_scree_pall.nc")


def logpdf(vec_slice, cov_slice):
    mean = np.zeros_like(vec_slice)
    return multivariate_normal.logpdf(vec_slice, mean=mean, cov=cov_slice)


ipdb.set_trace()

ll = xr.apply_ufunc(
    logpdf,
    scree["mean ulsa"],
    tensor["cov_pfreq"],
    input_core_dims=[["freqs2"], ["freqs2", "freqs2_"]],
    vectorize=True,
    output_dtypes=[np.float64],
)


##---------------------------------------------------------------------------##
# reload tensors

# NOTE: argparser needs something like: python simtensors.py --config configs/a{0..80..10}_dt3600.yaml
#
# # Load the lusee ulsa sim tensor
# print("Loading ULSA sim tensor..")
# parser = simutils.create_parser()
# args = parser.parse_args()
# sim = simutils.SimTensor().from_args(args)
# print("  saving as netcdf..")
# sim.to_netcdf("netcdf/sim.nc")

# # WARN: when loading netcdf tensor, ensure string coordniates are loaded as strings
# #
# print("loading netcdf sim tensor..")
# sim = xr.open_dataset("netcdf/sim.nc")
# sim["combs"] = sim["combs"].astype(str)
# ulsa = sim["ulsa"]

# # WARN: subtracting mean here, changes how some plots look
# # WARN: xarray svd does not rename dims correctly
# #
# print("hosvd..")
# print("  subtracting mean..")
# tensor = xr.Dataset({"ulsa": ulsa - ulsa.mean("times")})
# print("  unfolding..")
# tensor["f-mode"] = tensor["ulsa"].einops.rearrange("freqs (times angles combs)=tas")
# tensor["a-mode"] = tensor["ulsa"].einops.rearrange("angles (times freqs combs)=tfs")
# tensor["c-mode"] = tensor["ulsa"].einops.rearrange("combs (times freqs angles)=tfa")
# tensor["t-mode"] = tensor["ulsa"].einops.rearrange("times (freqs angles combs)=fac")
# print("  unfolded svd..")
# Uf, Sf, _ = tensor["f-mode"].linalg.svd(dims=["freqs", "tas"], full_matrices=False)
# Ua, Sa, _ = tensor["a-mode"].linalg.svd(dims=["angles", "tfs"], full_matrices=False)
# Uc, Sc, _ = tensor["c-mode"].linalg.svd(dims=["combs", "tfa"], full_matrices=False)
# Ut, St, _ = tensor["t-mode"].linalg.svd(dims=["times", "fac"], full_matrices=False)
# tensor["eig:freqs"] = Uf
# tensor["eig:angles"] = Ua
# tensor["eig:combs"] = Uc
# tensor["eig:times"] = Ut
# tensor["eva:freqs"] = xr.DataArray(Sf.data, dims=["freqs2"])
# tensor["eva:angles"] = xr.DataArray(Sa.data, dims=["angles2"])
# tensor["eva:combs"] = xr.DataArray(Sc.data, dims=["combs2"])
# tensor["eva:times"] = xr.DataArray(St.data, dims=["times2"])
# print("  calc core and core products..")
# tensor["core"] = Uf.T @ Ua.T @ Uc.T @ (Ut.T @ tensor["ulsa"])
# tensor["corexUt"] = tensor["core"] @ Ut
# tensor["corexUtxUc"] = tensor["core"] @ Ut @ Uc
# tensor["corexUtxUa"] = tensor["core"] @ Ut @ Ua
# tensor["corexUtxUf"] = tensor["core"] @ Ut @ Uf
# tensor["corexUtxUaxUc"] = tensor["core"] @ Ut @ Ua @ Uc
# tensor["corexUtxUfxUa"] = tensor["core"] @ Ut @ Uf @ Ua
# tensor["corexUtxUfxUc"] = tensor["core"] @ Ut @ Uf @ Uc
# tensor["corexUtxUfxUaxUc"] = tensor["core"] @ Ut @ Uf @ Ua @ Uc
# print("calc scree components..")
# tensor["ulsa_pfreq"] = tensor["eig:freqs"] @ sim["ulsa"]
# tensor["ulsa_pall"] = (
#     tensor["eig:freqs"] @ tensor["eig:angles"] @ tensor["eig:combs"] @ sim["ulsa"]
# )
# tensor["ulsa_nomean_pall"] = (
#     tensor["eig:freqs"] @ tensor["eig:angles"] @ tensor["eig:combs"] @ tensor["ulsa"]
# )
# tensor["ulsa_nomean_pfreq"] = tensor["eig:freqs"] @ tensor["ulsa"]
# print("calc covariances..")
# tensor["cov_pfreq"] = (
#     linalg.matmul(
#         tensor["ulsa_nomean_pfreq"],
#         tensor["ulsa_nomean_pfreq"],
#         dims=["freqs2", "times", "freqs2"],
#         out_append="_",
#     )
#     / 649
# )
# tensor["cov_pall"] = (
#     linalg.matmul(
#         tensor["ulsa_nomean_pall"],
#         tensor["ulsa_nomean_pall"],
#         dims=["freqs2", "times", "freqs2"],
#         out_append="_",
#     )
#     / 649
# )
# print("saving as netcdf..")
# tensor.to_netcdf("netcdf/sim_hosvd.nc")

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

# scree = xr.open_dataset("netcdf/sim_scree_pfreq.nc")
# scree["combs"] = scree["combs"].astype(str)
# pscree = xr.open_dataset("netcdf/sim_scree_pall.nc")

##---------------------------------------------------------------------------##

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

ipdb.set_trace()
