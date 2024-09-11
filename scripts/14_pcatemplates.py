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

import simflows.load as loader
import simflows.pca as simpca
import simflows.utils as simutils

##--------------------------------------------------------------------##
# %%
# load templates for ulsa, da, cmb

# templates = loader.load_templates(freqs=jnp.linspace(1, 50, 393), da_amp=4e6)
templates = loader.load_templates(da_amp=1)
mock = loader.load_mock_sim(da_amp=1)
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


# %%
