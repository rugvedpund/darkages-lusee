# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: lusee
#     language: python
#     name: lusee
# ---

import jax
import jax.numpy as jnp

# WARN: stupid jax doesn't like float64 by default
jax.config.update("jax_enable_x64", True)

import lusee
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xarray_einstats

import simflows.load as loader
import simflows.pca as simpca
import simflows.utils as simutils
import simflows.jax as simjax

# %%

"""
play with shapes
"""

# templates = loader.load_templates(freqs=jnp.linspace(1, 50, 393), da_amp=4e6)
templates = loader.load_templates(da_amp=1)
g = templates.plot.line(row="kind", sharey=False)
g.axs[0, 0].set_yscale("log")
plt.show()


# %%


ntimes = 650
idxs = simjax.random_normal((ntimes, 1), seed=42, mean=2.5, sigma=0.5)
amp30s = simjax.random_normal((ntimes, 1), seed=42, mean=1e5, sigma=1e6)
mock = loader.load_mock_sim(idxs=idxs, amp30s=amp30s, da_amp=1)
fg, da, cmb, sum, delta = mock

# %%


g=mock.sel(kind=['fg','delta']).mean('times').plot(col='kind',x="freqs", yscale='log')
plt.show()


# %%

# %%

plt.figure(figsize=(13, 3))
plt.subplot(131)
plt.plot(idxs)
plt.title('spectral index')
plt.subplot(132)
plt.semilogy(amp30s)
plt.title('amplitude')
plt.subplot(133)
fg.plot(norm=simutils.lognorm(),x="times")
plt.tight_layout()
plt.show()


# %%

mock.sel(kind=['fg','sum','delta']).plot(col='kind',x="times", norm=simutils.symlognorm())
plt.show()

# %%

g=mock.sel(kind=['fg','sum']).mean('times').plot(col='kind',x="freqs", yscale='log')
plt.show()

# %%

g=mock.sel(kind=['fg','delta']).mean('times').plot(col='kind',x="freqs", yscale='symlog')
plt.show()

# %%

plt.subplot(121)
fg.plot(norm=simutils.lognorm(),x="times")
plt.subplot(122)
mock['delta'].plot(x="times", norm=simutils.symlognorm())
plt.tight_layout()
plt.show()

# %%


# %%
