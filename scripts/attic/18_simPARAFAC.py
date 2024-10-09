import simutils
import simflows as sf
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import tensorly as tl
from tensorly.decomposition import parafac
import tlviz

##---------------------------------------------------------------------------##

parser = simutils.create_parser()
args = parser.parse_args()
sky = sf.SkyAnalyzer().from_args(args)
sky.set_comb(args.comb)

freqs = np.arange(1, 51)
times = np.arange(5850)
combs = np.arange(4)
# angles = [c.split("_")[0].split("/")[-1] for c in args.configs]

print(sky.ulsa.data.shape, sky.da.data.shape, sky.cmb.data.shape)

ulsa = np.dstack(
    [sky.ulsa.data[i * 50 : (i + 1) * 50, :].copy() for i in range(len(combs))]
)
da = np.dstack(
    [sky.da.data[i * 50 : (i + 1) * 50, :].copy() for i in range(len(combs))]
)
cmb = np.dstack(
    [sky.cmb.data[i * 50 : (i + 1) * 50, :].copy() for i in range(len(combs))]
)
print(ulsa.shape, da.shape, cmb.shape)

ulsa = xr.DataArray(ulsa,coords=[freqs, times, combs])
da = xr.DataArray(da,coords=[freqs, times, combs])
cmb = xr.DataArray(cmb,coords=[freqs, times, combs])

dataset = xr.Dataset(data_vars={"ulsa": ulsa, "da": da, "cmb": cmb})

U = tl.tensor(dataset.ulsa)
D = tl.tensor(dataset.da)
C = tl.tensor(dataset.cmb)

Ufac = parafac(U, 10, init="random", random_state=0)
fig, ax = tlviz.visualisation.components_plot(Ufac)
plt.show()

# tlviz.visualisation.core_element_heatmap(Ufac, U, annotate=False)
tlviz.visualisation.core_element_plot(Ufac, U)
plt.show()


# Dfac = parafac(D, 4, init="random", random_state=0)
# fig, ax = tlviz.visualisation.components_plot(Dfac)
# plt.show()

# Cfac = parafac(C, 4, init="random", random_state=0)
# fig, ax = tlviz.visualisation.components_plot(Cfac)
# plt.show()




##---------------------------------------------------------------------------##
# old shit


# rank = 4
# print("doing parafac for rank: ", rank, "..")
# weights,factors = parafac(tdata, rank=rank, n_iter_max=1000, normalize_factors=True)
## tFacs = [parafac(tdata, rank=rank, n_iter_max=1000, normalize_factors=True, tol=1e-10) for rank in ranks]
# widx = torch.argsort(weights,descending=True)
# weights = weights[widx]
# factors = [f[:,widx] for f in factors]

###---------------------------------------------------------------------------##

## def reconstructed_variance(tFac, tIn=None):
##     """ This function calculates the amount of variance captured (R2X) by the tensor method. """
##     tMask = np.isfinite(tIn)
##     vTop = np.sum(np.square(tl.cp_to_tensor(tFac) * tMask - np.nan_to_num(tIn)))
##     vBottom = np.sum(np.square(np.nan_to_num(tIn)))
##     return 1.0 - vTop / vBottom

## r2x = [reconstructed_variance(tFac, tIn=tdata) for tFac in tFacs]
## plt.plot(ranks, r2x, "bo")
## plt.xlabel("rank")
## plt.ylabel("variation explained")
## plt.show()

###---------------------------------------------------------------------------##

# fig,ax = plt.subplots(2,3,figsize=(12,8))
# ax = ax.flatten()


# for fi,f in enumerate(factors):
#    ax[fi].plot(f)

# ax[-1].plot(weights)
# ax[-1].set_title("weights")
# [ax[-1].plot([],[],label=f"compoment {i}",c=f"C{i}") for i in range(rank)]
# ax[-1].set_yscale("log")
# ax[-1].legend()

# plt.show()
