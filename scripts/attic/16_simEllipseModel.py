import simutils
import simflows as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import scipy
from copy import deepcopy


##---------------------------------------------------------------------------##

def linear(x, slope, intercept):
    return slope * x + intercept
def sqrt(x, amp):
    return amp * np.sqrt(x)

parser = simutils.create_parser()
args = parser.parse_args()
sky = sf.SkyAnalyzer().from_args(args)

combs_list = simutils.get_combs_list(args.comb)
gaussians = dict()
skies = dict()

for comb in combs_list:
    s=deepcopy(sky)
    s.set_comb(comb)
    s.doPCA_and_project()
    skies[comb] = deepcopy(s)
    gauss = sf.GaussianApprox()
    gauss.train(s)
    (slope, intercept), _ = scipy.optimize.curve_fit(linear, gauss.modes, gauss.sigmadev)
    # gauss.scale_model(linear(gauss.modes, slope=3*slope, intercept=3*intercept))
    gauss.scale_model(linear(gauss.modes, slope=3*slope, intercept=10.0))
    gaussians[comb] = gauss

true_amp = 1.0

##---------------------------------------------------------------------------##
# scree plot

fig, ax = plt.subplots(len(combs_list), 2, figsize=(12, 3*len(combs_list)))
for icomb, comb in enumerate(combs_list):
    simutils.plt_scree(skies[comb], ax=ax[icomb,0], true_amp=true_amp)
    ax[icomb,0].plot([], [], "C0", label="mean ulsa")
    ax[icomb,0].plot([], [], "C1", label="mean da")
    ax[icomb,0].plot([], [], "C2", label="mean cmb")
    ax[icomb,0].plot([], [], "C3", label="rms ulsa")

    ax[icomb,0].plot(gaussians[comb].sigmas * skies[comb].ulsa.proj_rms, c="C4", label="gauss", ls=":")
    ax[icomb,0].legend(loc="upper right")

    pmean = skies[comb].ulsa.proj_mean.copy()
    sigmadev = np.abs(pmean / skies[comb].ulsa.proj_rms.copy())
    ax[icomb,1].plot(sigmadev, c="C0", label="mean ulsa")
    ax[icomb,1].plot(np.ones_like(sigmadev), c="C3", label="rms ulsa")

    ax[icomb,1].plot(gaussians[comb].sigmas, c="C4", label="gauss", ls="--")
    ax[icomb,1].set_xlabel("eigmodes")
    ax[icomb,1].set_ylabel("sigma deviation")
    ax[icomb,1].legend(loc="upper right")

    ax[icomb,0].set_title(f"{comb}")
    ax[icomb,1].set_title(f"{comb}")
fig.tight_layout()
fig.show()

###---------------------------------------------------------------------------##
## Likelihood plot

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
da_amp = 40
cmb_amp = 2.7
print("true_amp: ", true_amp)

# amp = np.logspace(np.log10(true_amp*1e-2),np.log10(true_amp*1e5),1000)
# amp = np.linspace(-true_amp * 1e5, true_amp * 1e5, 1000)
amp = np.linspace(-true_amp * 1e5, true_amp * 1e5, 100000)
da_loglikelihoods = dict()
cmb_loglikelihoods = dict()
for comb, gauss in gaussians.items():
    print("calculating likelihoods for ", comb)
    da_loglikelihoods[comb]=gauss.get_da_loglikelihood(amp, true_amp=true_amp)
    cmb_loglikelihoods[comb]=gauss.get_cmb_loglikelihood(amp, true_amp=true_amp)
    ax[0].plot(amp, simutils.exp(da_loglikelihoods[comb]), label=comb, ls="--")
    ax[1].plot(amp, simutils.exp(cmb_loglikelihoods[comb]), label=comb, ls="--")

da_loglikelihoods["all"] = np.sum([da_loglikelihoods[comb] for comb in combs_list], axis=0)
cmb_loglikelihoods["all"] = np.sum([cmb_loglikelihoods[comb] for comb in combs_list], axis=0)
ax[0].plot(amp, simutils.exp(da_loglikelihoods["all"]), label="all", lw=2.0, color="k")
ax[1].plot(amp, simutils.exp(cmb_loglikelihoods["all"]), label="all", lw=2.0, color="k")

ax[0].axvline(x=da_amp*true_amp, color="gray", label="truth")
ax[0].axvline(x=-da_amp*true_amp, color="gray")
ax[1].axvline(x=cmb_amp*true_amp, color="gray", label="truth")
ax[1].axvline(x=-cmb_amp*true_amp, color="gray")

ax[0].set_ylabel("likelihood")
ax[1].set_ylabel("likelihood")
ax[0].set_xlabel("amplitude [mK]")
ax[1].set_xlabel("amplitude [K]")
ax[0].set_title(f"{true_amp:,.0f}x true DA amplitude")
ax[1].set_title(f"{true_amp:,.0f}x true CMB amplitude")
# ax[0].set_xscale("log")
# ax[1].set_xscale("log")
ax[0].legend()
ax[1].legend()
plt.suptitle(f"Product of GaussianApprox models: {args.comb}")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------##
# # pairplot


# def sns_add_rms_vline_diag(x, scale=1.0, **kwargs):
#     ax = plt.gca()
#     rms = np.sqrt(np.var(x))
#     rms *= scale
#     color = kwargs.get("color", "gray")
#     label = kwargs.get("label", None)
#     if rms > 0:
#         ax.axvline(rms, color=color, label=label)
#         ax.axvline(-rms, color=color, label=label)


# def sns_add_rms_ellipse_offdiag(x, y, scale=1.0, **kwargs):
#     ax = plt.gca()
#     color = kwargs.get("color", "gray")
#     label = kwargs.get("label", None)
#     if x.shape[0] < 2 or y.shape[0] < 2:
#         return
#     cov = np.cov(x, y)
#     lambda_, v = np.linalg.eig(cov)
#     lambda_ = np.sqrt(lambda_)
#     lambda_ *= scale
#     ell = mpl.patches.Ellipse(
#         xy=(np.mean(x), np.mean(y)),
#         width=lambda_[0] * 2,
#         height=lambda_[1] * 2,
#         angle=np.rad2deg(np.arccos(v[0, 0])),
#         label=label,
#     )
#     ell.set_facecolor("none")
#     ell.set_edgecolor(color)
#     ax.add_artist(ell)


# ndim, ndata = sky.ulsa.data.shape
# # eigmodes = np.arange(ndim)
# # eigmodes = [0, 10, 20, 30, 40]
# eigmodes = [0, 14, 29, 38, 49]
# print(eigmodes)
# print("plot pair plot..")
# pairplt = simutils.sns_pairplot(
#     eigmodes,
#     norm_pdata=sky.ulsa.norm_pdata,
#     ulsa_norm_pmean=sky.ulsa.norm_pmean,
#     da_norm_pmean=sky.da.norm_pmean * true_amp,
#     cmb_norm_pmean=sky.cmb.norm_pmean * true_amp,
# )
# pairplt = simutils.sns_pairplot_addGauss(pairplt, gaussB, eigmodes, color="C5")

# # pairplt = simutils.sns_pairplot_addVectors(
# #     pairplt,
# #     ulsa_norm_pmean=sky.ulsa.norm_pmean,
# #     sig_norm_pmean=sky.da.norm_pmean * true_amp,
# #     eigmodes=eigmodes,
# #     amp=amp,
# #     color="C2",
# # )
# # pairplt = simutils.sns_pairplot_addVectors(
# #     pairplt,
# #     ulsa_norm_pmean=sky.ulsa.norm_pmean,
# #     sig_norm_pmean=sky.cmb.norm_pmean * true_amp,
# #     eigmodes=eigmodes,
# #     amp=amp,
# #     color="C3",
# # )

# pairplt.map_diag(sns_add_rms_vline_diag, color="C0")
# pairplt.map_offdiag(sns_add_rms_ellipse_offdiag, color="C0")

# # sigma_scale = sigmamaxdev
# # pairplt.map_diag(sns_add_rms_vline_diag, color="r", scale=sigma_scale)
# # pairplt.map_offdiag(sns_add_rms_ellipse_offdiag, color="r", scale=sigma_scale)
# # plt.savefig(f"outputs/pairplot_{args.comb}_a0-80_gauss.png")
# # plt.savefig(f"outputs/pairplot_{args.comb}_a0-80_gauss.png")

# plt.show()


##---------------------------------------------------------------------------##
# correlation matrix

# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# corr = np.corrcoef(sky.ulsa.norm_pdata)
# im=sns.heatmap(corr - np.diag(np.diag(corr)), ax=ax, square=True, center=0)
# im.invert_yaxis()
# plt.title("correlation matrix of ulsa norm pdata")
# plt.show()
