# %%

import jax
import jax.numpy as jnp

# WARN: stupid jax doesn't like float64 by default
jax.config.update("jax_enable_x64", True)

import lusee
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xarray_einstats

import simflows.load as simloader
import simflows.pca as simpca
import simflows.utils as simutils

##--------------------------------------------------------------------##
# %%
# load templates for ulsa, da, cmb

# templates = loader.load_templates(freqs=jnp.linspace(1, 50, 393), da_amp=4e6)
templates = simloader.load_templates(da_amp=1)
mock = simloader.load_mock_sim(da_amp=1)
fg, da, cmb, noise = mock

g = templates.plot.line(row="kind", sharey=False)
g.axs[0, 0].set_yscale("log")
plt.show()

plt.figure(figsize=(13, 3))
plt.subplot(131)
fg.plot(norm=simutils.lognorm())
plt.subplot(132)
da.plot()
plt.subplot(133)
cmb.plot()
plt.show()
noise.plot()
plt.show()


# %%

# %%
##--------------------------------------------------------------------##
# %%
"""
┌─────────────────────────────────┐
│ individual pca of the mock sims │
└─────────────────────────────────┘
"""

mpca = simpca.get_pca(mock, "times", "freqs", other_dims=[])

mpca.S.plot(col="kind", yscale="log")
plt.show()

mpca.U.sel(
    freqs_eig=[
        1.0,
        2.0,
        3.0,
    ]
).plot.line(col="kind", x="freqs")
plt.show()

# mpca.Vt.plot(col="kind")
# plt.show()

# %%
##--------------------------------------------------------------------##
# %%

sum = mock.sum("kind")
sum.mean("times").plot(yscale="log")
plt.show()

spca = simpca.get_pca(sum, "times", "freqs", [])

spca.S.plot(yscale="log")
plt.show()

spca.U.sel(
    freqs_eig=[
        1.0,
        2.0,
        3.0,
    ]
).plot.line(x="freqs")
plt.show()

# spca.Vt.plot()
# plt.show()
# %%

spca.U.plot()
plt.show()

spca.Vt.plot()
plt.show()

# spca.to_dataarray(dim='pca').plot(col='pca')
# plt.show()


# ## analyze mock fg shapes using PCA
#
# ## status
# - [ ] TODO: task 1: description of the first task.
# - [ ] FIX: task 2: description of the second task.
# - [x] task 3: description of the third task.
#
# ## notes
# - important note 1.
# - WARN: important note 2.
# - BUG: important note 3.

# ---

from simflows.imports import *

ntimes = 650
idxs = simjax.random_normal((ntimes, 1), seed=0, mean=2.5, sigma=0.5)
amp20MHz = simjax.random_normal((ntimes, 1), seed=1, mean=1e4, sigma=1e4)
amp20MHz = jnp.abs(amp20MHz)
mock = simloader.load_mock_sim(idxs=idxs, amp20MHz=amp20MHz, da_amp=1)
fg, da, cmb, sum, delta = mock

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(idxs, amp20MHz)
ax[0].set_xlabel("spectral index")
ax[0].set_ylabel("amplitude at 20MHz")
fg.plot(
    x="freqs",
    hue="times",
    add_legend=False,
    yscale="log",
    color="grey",
    alpha=0.1,
    ax=ax[1],
)
ax[1].errorbar(fg.freqs, fg.mean("times"), fg.std("times"), fmt="o-", color="C0")
ax[1].set_yscale("symlog")
plt.show()


# ---

# g=mock.sel(kind=['fg','delta']).mean('times').plot(col='kind',x="freqs", yscale='log')
# plt.show()

# ---

plt.figure(figsize=(13, 3))
plt.subplot(131)
plt.plot(idxs)
plt.title("spectral index")
plt.subplot(132)
plt.semilogy(amp20MHz)
plt.title("amplitude")
plt.subplot(133)
fg.plot(norm=simutils.lognorm(), x="times")
plt.tight_layout()
plt.show()


# ---

mock.sel(kind=["fg", "sum", "delta"]).plot(
    col="kind", x="times", norm=simutils.symlognorm()
)
plt.show()

g = mock.sel(kind=["fg", "sum"]).mean("times").plot(col="kind", x="freqs", yscale="log")
plt.show()

g = (
    mock.sel(kind=["fg", "delta"])
    .mean("times")
    .plot(col="kind", x="freqs", yscale="symlog")
)
plt.show()

plt.subplot(121)
fg.plot(norm=simutils.lognorm(), x="times")
plt.subplot(122)
mock["delta"].plot(x="times", norm=simutils.symlognorm())
plt.tight_layout()
plt.show()


# %%
