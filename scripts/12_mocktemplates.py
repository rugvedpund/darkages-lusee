# %%

import jax
import jax.numpy as jnp

# WARN: stupid jax doesn't like float64 by default
jax.config.update("jax_enable_x64", True)

import lusee
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import simflows.load as loader
import simflows.utils as simutils

##--------------------------------------------------------------------##
# %%
# load templates for ulsa, da, cmb

templates = loader.load_templates(freqs=jnp.linspace(1, 50, 393), da_amp=4e6)
mock = loader.load_mock_sim()


g = templates.plot.line(row="kind", sharey=False)
g.axs[0, 0].set_yscale("log")
plt.show()
g

##--------------------------------------------------------------------##
# %%
"""
┌────────────────────────────────────────────────────────────────────────┐
│ taking ratios makes all temperatures positive, intuitively should help │
└────────────────────────────────────────────────────────────────────────┘
"""

ratio = templates / templates.sel(freqs=30)
g = ratio.plot.line(row="kind", sharey=False)
g.axs[0, 0].set_yscale("log")
plt.show()
g

##--------------------------------------------------------------------##
# %%
# what happens to the sum
# NOTE: the sum is of ratios is incorrect, since the denominators are not the same

sum = templates.sum("kind")
# sum /= sum.sel(freqs=30)
dsumdnu = xr.DataArray(np.gradient(sum), dims=sum.dims, coords=sum.coords)

sum.plot(yscale="log")
dsumdnu.plot(yscale="symlog")
# plt.xlim(10,50)
plt.show()


# %%

ratio = templates / templates.sel(freqs=30)
g = ratio.plot.line(row="kind", sharey=False)
g.axs[0, 0].set_yscale("log")
plt.show()
g

# %%

##--------------------------------------------------------------------##
# %%


# %%
