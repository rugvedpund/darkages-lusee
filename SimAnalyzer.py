import lusee
import numpy as np
import NormalizingFlow as nf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

def all_combs(n):
    combs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                combs.append(f"{i}{j}R")
            if i > j:
                combs.append(f"{j}{i}R")
            if i < j:
                combs.append(f"{i}{j}I")
    return combs

def plt_waterfall(D, comb, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(
        D[:, comb, :].T,
        aspect="auto",
        # extent=(D.freq[0], D.freq[-1], len(D.times), 0),
        origin="lower",
        cmap="viridis",
        **kwargs
    )
    cbar=plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.ax.tick_params(labelsize=8)
    ax.set_ylabel("f(MHz)")
    ax.set_xlabel("time")
    ax.set_title(f"{comb}")
    return ax

D = lusee.Data("gaussbeam.fits")
fig, ax = plt.subplots(4, 4, figsize=(15, 8))
for comb, ax in zip(all_combs(4), ax.flatten()):
    plt_waterfall(D, comb, ax=ax, norm=mcolors.SymLogNorm(linthresh=1e2))
fig.suptitle(r"NSEW Gaussian $10^\circ$ Beams at $40^\circ$ declination")
fig.tight_layout()
plt.show()

D = lusee.Data("realistic_example.fits")
fig, ax = plt.subplots(4, 4, figsize=(15, 8))
for comb, ax in zip(all_combs(4), ax.flatten()):
    plt_waterfall(D, comb, ax=ax, norm=mcolors.SymLogNorm(linthresh=1e2))
fig.suptitle("NSEW LuSEE Beams")
fig.tight_layout()
plt.show()

