import fitsio
import healpy as hp
import ipdb
import lusee
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from simutils import MollviewAccessor
from xarray_einstats import einops, linalg

log = mpl.colors.LogNorm()
symlog = mpl.colors.SymLogNorm(linthresh=1e2)


def t21da(xarr_dim_sigma_freq: xr.DataArray):
    base_freqs = np.arange(1, 51)
    base_t21 = lusee.SkyModels.T_DarkAges(base_freqs)
    t21 = np.zeros_like(xarr_dim_sigma_freq).astype("float64")
    t21[:50] = base_t21
    return xr.DataArray(
        t21, coords=xarr_dim_sigma_freq.coords, dims=xarr_dim_sigma_freq.dims
    )


def cmb(xarr_dim_sigma_freq: xr.DataArray):
    base_freqs = np.arange(1, 51)
    base_cmb = 2.73 * np.ones_like(base_freqs)
    cmb = np.zeros_like(xarr_dim_sigma_freq).astype("float64")
    cmb[:50] = base_cmb
    return xr.DataArray(
        cmb, coords=xarr_dim_sigma_freq.coords, dims=xarr_dim_sigma_freq.dims
    )


print("loading ulsa fits..")
ulsa = fitsio.read("ulsa.fits")
nfreq, npix = ulsa.shape
freqs = np.linspace(1, 50, nfreq)
pixels = np.arange(npix)

# hp.mollview(ulsa[19,:],title='ULSA @20MHz',unit="T[K]");plt.savefig("./tex/figures/ulsa20MHz.pdf")
# plt.imshow(ulsa,aspect="auto",norm=log);plt.colorbar(label="T[K]");plt.xlabel("healpix pix (nside=128)");plt.ylabel("freq [MHz]");plt.title('ULSA 1-50MHz');plt.savefig("./tex/figures/ulsa1-50MHz.pdf")

print("smoothing..")
sigmas = [2, 4, 6]
smoothing = []
for i, sigma in enumerate(sigmas):
    # NOTE: smooth over sigmas by looping over frequencies because hp.smoothing is stupid
    smoothed = np.vstack(
        [
            hp.sphtfunc.smoothing(ulsa[f, :], sigma=np.deg2rad(sigma))
            for f, _ in enumerate(freqs)
        ]
    )
    smoothing.append(smoothed)

# calculate hosvd
# NOTE: careful with the shape of Up, since we have full_matrices = False
print("hosvd..")
# WARN: this changes results
smooth_ulsa246 = np.array([smoothing[0], smoothing[1], smoothing[2]])
# smooth_ulsa246 = np.array(
#     [smoothing[0], smoothing[1] - smoothing[0], smoothing[2] - smoothing[1]]
# )
xarr246 = xr.DataArray(
    smooth_ulsa246, coords=[sigmas, freqs, pixels], dims=["sigma", "freq", "pix"]
)
# WARN: this changes results
xarr246 = xarr246 - xarr246.mean(dim="pix")
tensor246 = xr.Dataset({"ulsa": xarr246})
print("  unfolding..")
tensor246["f-mode"] = tensor246["ulsa"].einops.rearrange("freq (sigma pix)=sigma_pix")
tensor246["s-mode"] = tensor246["ulsa"].einops.rearrange("sigma (freq pix)=freq_pix")
tensor246["p-mode"] = tensor246["ulsa"].einops.rearrange("pix (sigma freq)=pix2")
print("  unfolded svd..")
Uf, Sf, _ = tensor246["f-mode"].linalg.svd(
    dims=["freq", "sigma_pix"], full_matrices=False
)
Us, Ss, _ = tensor246["s-mode"].linalg.svd(
    dims=["sigma", "freq_pix"], full_matrices=False
)
Up, Sp, _ = tensor246["p-mode"].linalg.svd(dims=["pix", "pix2"], full_matrices=False)
print("  calc core and core products..")
tensor246["core"] = Uf.T @ (Us.T @ Up.T @ tensor246["ulsa"])
tensor246["corexUp"] = tensor246["core"] @ Up
tensor246["corexUpxUs"] = tensor246["core"] @ Up @ Us
tensor246["corexUpxUf"] = tensor246["core"] @ Up @ Uf
tensor246["corexUpxUsxUf"] = tensor246["core"] @ Up @ Us @ Uf
print("tensor246 ready")

# ipdb.set_trace()

# # make projected plots
# print('saving large plots..')
# tensor246.mollview.plot('corexUp',sel=dict(freq2=np.arange(50)),col='sigma2',row='freq2',sharex=False,sharey=False)
# plt.savefig("./figures/corexUp.png")
# tensor246.mollview.plot(
#     "corexUpxUs",
#     sel=dict(freq2=np.arange(50)),
#     col="sigma",
#     row="freq2",
#     sharex=False,
#     sharey=False,
# )
# plt.savefig("./figures/corexUpxUs.png")
# tensor246.mollview.plot('corexUpxUsxUf',sel=dict(freq=np.arange(1,51)),col='sigma',row='freq',sharex=False,sharey=False)
# plt.savefig("./figures/corexUpxUsxUf.png")
# tensor246.mollview.plot(
#     "corexUpxUf",
#     sel=dict(freq=np.arange(1, 51)),
#     col="sigma2",
#     row="freq",
#     sharex=False,
#     sharey=False,
# )
# plt.savefig("./figures/corexUpxUf.png")
# # plot sigmas and freqs
# tensor246.mollview.plot('ulsa',sel=dict(freq=[10,20,30,40]),col='sigma',row='freq',sharex=False,sharey=False)
# plt.show()


# accesses SimTensorAccessor
print("creating xarray..")
# WARN: this changes results
# smooth_ulsa246 = np.array([smoothing[0], smoothing[1], smoothing[2]])
smooth_ulsa246 = np.array(
    [smoothing[0], smoothing[1] - smoothing[0], smoothing[2] - smoothing[1]]
)
xarr = xr.DataArray(
    smooth_ulsa246, coords=[sigmas, freqs, pixels], dims=["sigma", "freq", "pix"]
)
# WARN: this changes results
xarr = xarr - xarr.mean(dim="pix")
tensor = xr.Dataset({"ulsa": xarr})
tensor["ulsa2"] = (
    tensor["ulsa"].sel(sigma=[2]).einops.rearrange("(sigma freq)=sigma_freq2 pix")
)
tensor["ulsa24"] = (
    tensor["ulsa"].sel(sigma=[2, 4]).einops.rearrange("(sigma freq)=sigma_freq24 pix")
)
tensor["ulsa246"] = (
    tensor["ulsa"]
    .sel(sigma=[2, 4, 6])
    .einops.rearrange("(sigma freq)=sigma_freq246 pix")
)
tensor["da2"] = t21da(tensor["ulsa2"].sigma_freq2)
tensor["da24"] = t21da(tensor["ulsa24"].sigma_freq24)
tensor["da246"] = t21da(tensor["ulsa246"].sigma_freq246)
tensor["cmb2"] = cmb(tensor["ulsa2"].sigma_freq2)
tensor["cmb24"] = cmb(tensor["ulsa24"].sigma_freq24)
tensor["cmb246"] = cmb(tensor["ulsa246"].sigma_freq246)

# # plot sigmas and freqs
# tensor.mollview.plot('ulsa',sel=dict(freq=[10,20,30,40]),col='sigma',row='freq',sharex=False,sharey=False)
# g.fig.suptitle(r'ULSA: 10,20,30,40MHz and $2^\circ, 4^\circ, 6^\circ$ beams')
# plt.show()

# calculate svd
print("svd..")
svd = xr.Dataset()
for map in ["2", "24", "246"]:
    ulsamap, p_ulsamap, p_damap, mapdim = (
        f"ulsa{map}",
        f"p_ulsa{map}",
        f"p_da{map}",
        f"sigma_freq{map}",
    )
    p_cmbmap = f"p_cmb{map}"
    U, S, Vt = tensor[ulsamap].linalg.svd(dims=[mapdim, "pix"], full_matrices=False)
    svd["ulsa" + map] = xr.DataArray({"U": U, "S": S, "Vt": Vt})
    tensor[p_ulsamap] = linalg.matmul(
        U, tensor[ulsamap], dims=[[f"{mapdim}2", mapdim], [mapdim, "pix"]]
    )
    tensor[p_ulsamap] = U.T @ tensor[ulsamap]
    tensor[p_damap] = U.T @ tensor["da" + map]
    tensor[p_cmbmap] = U.T @ tensor["cmb" + map]
    tensor["p_rms" + map] = np.sqrt(tensor["p_ulsa" + map].var(dim="pix"))
    tensor["norm_pulsamean" + map] = (
        tensor["p_ulsa" + map].mean(dim="pix") / tensor["p_rms" + map]
    )
    tensor["norm_pulsa" + map] = tensor["p_ulsa" + map] / tensor["p_rms" + map]
    tensor["norm_pda" + map] = tensor["p_da" + map] / tensor["p_rms" + map]
    tensor["norm_pcmb" + map] = tensor["p_cmb" + map] / tensor["p_rms" + map]
print("tensor ready")

ipdb.set_trace()

# # save large pairplots
# # FIX: this does not work, xarray -> numpy -> pandas -> seaborn is stupid
#
# print('saving large pairplots..')
# modes = [0,5,10,30,45]
# print('ulsa2')
# simutils.sns_pairplot(modes,norm_pdata=tensor['norm_pulsa2'],ulsa_norm_pmean=tensor['norm_pulsamean2'],da_norm_pmean=tensor['norm_pda2'],cmb_norm_pmean=tensor['norm_pcmb2'])
# plt.savefig("./figures/pairplot_ulsa2.png")
# print('ulsa24')
# simutils.sns_pairplot(modes,norm_pdata=tensor['norm_pulsa24'],ulsa_norm_pmean=tensor['norm_pulsamean24'],da_norm_pmean=tensor['norm_pda24'],cmb_norm_pmean=tensor['norm_pcmb24'])
# plt.savefig("./figures/pairplot_ulsa24.png")
# print('ulsa246')
# simutils.sns_pairplot(modes,norm_pdata=tensor['norm_pulsa246'],ulsa_norm_pmean=tensor['norm_pulsamean246'],da_norm_pmean=tensor['norm_pda246'],cmb_norm_pmean=tensor['norm_pcmb246'])
# plt.savefig("./figures/pairplot_ulsa246.png")


# # plot projected tensor
# tensor.mollview.plot('p_ulsa2',sel=dict(sigma_freq22=[0,1,2,4,5]),row='sigma_freq22',sharex=False,sharey=False);plt.show()
# tensor.mollview.plot('p_ulsa24',sel=dict(sigma_freq242=[0,1,2,4,5]),row='sigma_freq242',sharex=False,sharey=False);plt.show()
# tensor.mollview.plot('p_ulsa246',sel=dict(sigma_freq2462=[0,1,2,4,5]),row='sigma_freq2462',sharex=False,sharey=False);plt.show()

# # plot projected tensor
# print('saving large plots..')
# tensor.mollview.plot('p_ulsa2',sel=dict(sigma_freq22=np.arange(50)),row='sigma_freq22',sharex=False,sharey=False)
# plt.savefig("./figures/ulsa2.png")
# tensor.mollview.plot('p_ulsa24',sel=dict(sigma_freq242=np.arange(50)),row='sigma_freq242',sharex=False,sharey=False)
# plt.savefig("./figures/ulsa24.png")
# tensor.mollview.plot('p_ulsa246',sel=dict(sigma_freq2462=np.arange(50)),row='sigma_freq2462',sharex=False,sharey=False)
# plt.savefig("./figures/ulsa246.png")
