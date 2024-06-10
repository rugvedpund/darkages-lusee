import simutils
import simflows as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl


def sns_add_rms_vline_diag(x, scale=1.0, **kwargs):
    ax = plt.gca()
    rms = np.sqrt(np.var(x))
    rms *= scale
    color = kwargs.get('color', 'gray')
    label = kwargs.get('label', None)
    if rms > 0:
        ax.axvline(rms, color=color, label=label)
        ax.axvline(-rms, color=color, label=label)

def sns_add_rms_ellipse_offdiag(x, y, scale=1.0, **kwargs):
    ax = plt.gca()
    color = kwargs.get('color', 'gray')
    label = kwargs.get('label', None)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    lambda_ *= scale
    ell = mpl.patches.Ellipse(xy=(np.mean(x), np.mean(y)),
                              width=lambda_[0]*2, height=lambda_[1]*2,
                              angle=np.rad2deg(np.arccos(v[0, 0])),
                              label=label)
    ell.set_facecolor('none')
    ell.set_edgecolor(color)
    ax.add_artist(ell)
##---------------------------------------------------------------------------##

parser = simutils.create_parser()
args = parser.parse_args()
sky = sf.SkyAnalyzer().from_args(args)
sky.set_comb(args.comb)
sky.doPCA_and_project()

cov = np.cov(sky.ulsa.norm_pdata)
mu = np.mean(sky.ulsa.norm_pdata, axis=1)
pmean = sky.ulsa.norm_pmean.copy()

sigmadev = np.abs(pmean - mu) / np.sqrt(np.diag(cov))
sigmamaxdev = np.max(sigmadev)
sigmaavgdev = np.mean(sigmadev)
print(f"sigmaavgdev: {sigmaavgdev:.2f}, sigmamaxdev: {sigmamaxdev:.2f}")

##---------------------------------------------------------------------------##
#correlation matrix
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
corr = np.corrcoef(sky.ulsa.norm_pdata)
im=sns.heatmap(corr - np.diag(np.diag(corr)), ax=ax, square=True, center=0)
im.invert_yaxis()
plt.title("correlation matrix of ulsa norm pdata")
plt.show()

##---------------------------------------------------------------------------##
#scree plot
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
simutils.plt_scree(sky, ax=ax[0])
ax[0].plot(sky.ulsa.proj_rms*sigmamaxdev, c="C3", ls=":", label=f"{sigmamaxdev:.1f}x scaled rms ulsa")
ax[0].plot(sky.ulsa.proj_rms*sigmaavgdev, c="C3", ls="--", label=f"{sigmaavgdev:.1f}x scaled rms ulsa")

ax[0].plot([], [], "C0", label="mean ulsa")
ax[0].plot([], [], "C1", label="mean da")
ax[0].plot([], [], "C2", label="mean cmb")
ax[0].plot([], [], "C3", label="rms ulsa")
ax[0].legend()

ax[1].plot(sigmadev, c="C0", label="mean ulsa")
ax[1].plot(np.sqrt(np.diag(cov)), c="C3", label="rms ulsa")
ax[1].plot(range(len(sigmadev)), np.ones_like(sigmadev)*sigmamaxdev, c="C3", ls=":", label=f"{sigmamaxdev:.1f}x scaled rms ulsa")
ax[1].plot(range(len(sigmadev)), np.ones_like(sigmadev)*sigmaavgdev, c="C3", ls="--", label=f"{sigmaavgdev:.1f}x scaled rms ulsa")
ax[1].set_xlabel("eigmodes")
ax[1].set_ylabel("sigma deviation")
ax[1].legend()

plt.show()
##---------------------------------------------------------------------------##

#pairplot
ndim, ndata = sky.ulsa.data.shape
# eigmodes = np.arange(ndim)
# eigmodes = [0, 10, 20, 30, 40]
eigmodes = [0,14,29,38,49]
print(eigmodes)
print("plot pair plot..")
pairplt = simutils.sns_pairplot(eigmodes, sky=sky)
pairplt.map_diag(sns_add_rms_vline_diag, color="C0")
pairplt.map_offdiag(sns_add_rms_ellipse_offdiag, color="C0")
sigma_scale = sigmamaxdev
pairplt.map_diag(sns_add_rms_vline_diag, color="r", scale=sigma_scale)
pairplt.map_offdiag(sns_add_rms_ellipse_offdiag, color="r", scale=sigma_scale)
# plt.savefig(f"outputs/pairplot_{args.comb}_a0-80_gauss.png")
plt.show()

##---------------------------------------------------------------------------##
true_amp = 1.0
amp = np.logspace(np.log10(true_amp*1e-2),np.log10(true_amp*100),10000)[:,None]
gauss = sf.GaussianApprox().train(sky.ulsa.norm_pdata.T)
ulsa_means = sky.ulsa.norm_pmean.copy()[None,:]
da_means = sky.da.norm_pmean.copy()[None,:] * true_amp
cmb_means = sky.cmb.norm_pmean.copy()[None,:] * true_amp
ampxda_means = sky.da.norm_pmean.copy()[None,:] * amp
ampcmb_means = sky.cmb.norm_pmean.copy()[None,:] * amp
da_loglikelihood = gauss.loglikelihood(ampxda_means)
cmb_loglikelihood = gauss.loglikelihood(ampcmb_means)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(amp, simutils.exp(da_loglikelihood))
ax[1].plot(amp, simutils.exp(cmb_loglikelihood))
ax[0].axvline(x=true_amp, linestyle="--", color="k", label="truth")
ax[1].axvline(x=true_amp, linestyle="--", color="k", label="truth")
ax[0].set_ylabel("likelihood")
ax[1].set_ylabel("likelihood")
ax[0].set_xlabel("test amplitude")
ax[1].set_xlabel("test amplitude")
ax[0].set_title(f"{true_amp:,.0f}x true DA amplitude") 
ax[1].set_title(f"{true_amp:,.0f}x true CMB amplitude")
ax[0].set_xscale("log")
ax[1].set_xscale("log")
plt.suptitle(f"GaussianApprox model: {args.comb}")
plt.tight_layout()
plt.show()


