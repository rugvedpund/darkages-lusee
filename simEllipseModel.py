import simutils
import simflows as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import scipy


##---------------------------------------------------------------------------##

parser = simutils.create_parser()
args = parser.parse_args()
sky = sf.SkyAnalyzer().from_args(args)
sky.set_comb(args.comb)
sky.doPCA_and_project()

pmean = sky.ulsa.proj_mean.copy()
sigmadev = np.abs(pmean / sky.ulsa.proj_rms.copy())

gaussRef = sf.GaussianApprox()
gaussRef.train(sky)


def linear(x, slope, intercept):
    return slope * x + intercept


def sqrt(x, amp):
    return amp * np.sqrt(x)


gaussA = sf.GaussianApprox()
gaussA.train(sky)
sigmamaxdev = np.max(gaussA.sigmadev)
gaussA.scale_model(linear(gaussA.modes, slope=0.0, intercept=sigmamaxdev))

gaussB = sf.GaussianApprox()
gaussB.train(sky)
gaussB.scale_model(linear(gaussB.modes, slope=1.0, intercept=1.0))

gaussC = sf.GaussianApprox()
gaussC.train(sky)
(slope, intercept), _ = scipy.optimize.curve_fit(linear, gaussC.modes, gaussC.sigmadev)
gaussC.scale_model(linear(gaussC.modes, slope=3*slope, intercept=3*intercept))

gaussD = sf.GaussianApprox()
gaussD.train(sky)
# amp,_ = scipy.optimize.curve_fit(sqrt, gaussD.modes, gaussD.sigmadev)
gaussD.scale_model(sqrt(gaussD.modes, amp=10.0))

true_amp = 1.0

##---------------------------------------------------------------------------##
# scree plot

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
simutils.plt_scree(sky, ax=ax[0], true_amp=true_amp)
ax[0].plot([], [], "C0", label="mean ulsa")
ax[0].plot([], [], "C1", label="mean da")
ax[0].plot([], [], "C2", label="mean cmb")
ax[0].plot([], [], "C3", label="rms ulsa")

ax[0].plot(gaussRef.sigmas * sky.ulsa.proj_rms, c="C4", label="gauss", ls="--")
ax[0].plot(gaussA.sigmas * sky.ulsa.proj_rms, c="C5", label=f"A")
ax[0].plot(gaussB.sigmas * sky.ulsa.proj_rms, c="C5", label="B", ls="--")
ax[0].plot(gaussC.sigmas * sky.ulsa.proj_rms, c="C5", label="C", ls=":")
ax[0].plot(gaussD.sigmas * sky.ulsa.proj_rms, c="C5", label="D", ls="-.")
ax[0].legend(loc="upper right")

ax[1].plot(sigmadev, c="C0", label="mean ulsa")
ax[1].plot(np.ones_like(sigmadev), c="C3", label="rms ulsa")

ax[1].plot(gaussRef.sigmas, c="C4", label="gauss", ls="--")
ax[1].plot(gaussA.sigmas, c="C5", label="A")
ax[1].plot(gaussB.sigmas, c="C5", label="B", ls="--")
ax[1].plot(gaussC.sigmas, c="C5", label="C", ls=":")
ax[1].plot(gaussD.sigmas, c="C5", label="D", ls="-.")
ax[1].set_xlabel("eigmodes")
ax[1].set_ylabel("sigma deviation")
ax[1].legend(loc="upper right")

plt.show()

###---------------------------------------------------------------------------##
## Likelihood plot

# amp = np.logspace(np.log10(true_amp*1e-2),np.log10(true_amp*1e5),1000)
# amp = np.linspace(-true_amp * 1e5, true_amp * 1e5, 1000)
amp = np.linspace(-true_amp * 1e3, true_amp * 1e3, 1000)

llda_ref = gaussRef.get_da_loglikelihood(amp, true_amp=true_amp)
llda_A = gaussA.get_da_loglikelihood(amp, true_amp=true_amp)
llda_B = gaussB.get_da_loglikelihood(amp, true_amp=true_amp)
llda_C = gaussC.get_da_loglikelihood(amp, true_amp=true_amp)
llda_D = gaussD.get_da_loglikelihood(amp, true_amp=true_amp)

llcmb_ref = gaussRef.get_cmb_loglikelihood(amp, true_amp=true_amp)
llcmb_A = gaussA.get_cmb_loglikelihood(amp, true_amp=true_amp)
llcmb_B = gaussB.get_cmb_loglikelihood(amp, true_amp=true_amp)
llcmb_C = gaussC.get_cmb_loglikelihood(amp, true_amp=true_amp)
llcmb_D = gaussD.get_cmb_loglikelihood(amp, true_amp=true_amp)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
da_amp = 40
cmb_amp = 2.7

ax[0].plot(da_amp*amp, simutils.exp(llda_ref),c="C4", label="gauss")
ax[0].plot(da_amp*amp, simutils.exp(llda_A),c="C5", label="A")
ax[0].plot(da_amp*amp, simutils.exp(llda_B),c="C5", label="B", ls="--")
ax[0].plot(da_amp*amp, simutils.exp(llda_C),c="C5", label="C", ls=":")
ax[0].plot(da_amp*amp, simutils.exp(llda_D),c="C5", label="D", ls="-.")

ax[1].plot(cmb_amp*amp, simutils.exp(llcmb_ref),c="C4", label="gauss")
ax[1].plot(cmb_amp*amp, simutils.exp(llcmb_A),c="C5", label="A")
ax[1].plot(cmb_amp*amp, simutils.exp(llcmb_B),c="C5", label="B", ls="--")
ax[1].plot(cmb_amp*amp, simutils.exp(llcmb_C),c="C5", label="C", ls=":")
ax[1].plot(cmb_amp*amp, simutils.exp(llcmb_D),c="C5", label="D", ls="-.")

ax[0].axvline(x=da_amp*true_amp, color="k", label="truth", lw=0.5)
ax[1].axvline(x=cmb_amp*true_amp, color="k", label="truth", lw=0.5)
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
plt.suptitle(f"GaussianApprox model: {args.comb}")
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
