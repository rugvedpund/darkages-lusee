import ipdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import simutils
import xarray as xr
from xarray_einstats import einops, linalg

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

# # Load the lusee ulsa sim tensor
# # NOTE: argparser needs something like: python simtensors.py --config configs/a{0..80..10}_dt3600.yaml
#
# print("Loading ULSA sim tensor..")
# parser = simutils.create_parser()
# args = parser.parse_args()
# sim = simutils.SimTensor().from_args(args)

# # NOTE: sim products angles 0-80, all combs, 50 freq bins and 650 timesteps ~1h
#
# print("loading netcdf sim tensor..")
# sim = xr.open_dataset("netcdf/sim.nc")
# ulsa = sim["ulsa"]
# print("hosvd..")
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
# print("  calc core and core products..")
# tensor["core"] = Uf.T @ Ua.T @ Uc.T @ (Ut.T @ tensor["ulsa"])
# tensor["corexUt"] = tensor["core"] @ Ut
# tensor["corexUtxUc"] = tensor["core"] @ Ut @ Uc
# tensor["corexUtxUa"] = tensor["core"] @ Ut @ Ua
# tensor["corexUtxUf"] = tensor["core"] @ Ut @ Uf
# tensor["corexUtxUfxUa"] = tensor["core"] @ Ut @ Uf @ Ua
# tensor["corexUtxUfxUc"] = tensor["core"] @ Ut @ Uf @ Uc
# tensor["corexUtxUfxUaxUc"] = tensor["core"] @ Ut @ Uf @ Ua @ Uc
# # print("  saving as netcdf..")
# # tensor.to_netcdf("netcdf/sim_hosvd.nc")

tensor = xr.open_dataset("netcdf/sim_hosvd.nc")
ulsa = tensor["ulsa"]

# # plot all combs, single angle
# ulsa.sel(angles=0,combs=combmat).plot.imshow(col='combs',col_wrap=4,norm=symlog,cmap='viridis',y='freqs',robust=True)
# plt.show()
# # plt.savefig('tex/figures/sim0.pdf')

# # # plot all angles, single comb
# ulsa.sel(combs="00R").plot.pcolormesh(
#     col="angles", col_wrap=3, y="freqs", aspect=2.0, norm=symlog, cmap="viridis"
# )
# plt.show
# # plt.savefig('tex/figures/simC00R.pdf')

ipdb.set_trace()
