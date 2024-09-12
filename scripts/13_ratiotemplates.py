# %%

import jax
import jax.numpy as jnp

# WARN: stupid jax doesn't like float64 by default
jax.config.update("jax_enable_x64", True)

import lusee
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import simflows.load as simloader
import simflows.utils as simutils

##--------------------------------------------------------------------##
# %%
# load templates for ulsa, da, cmb

templates = simloader.load_templates(freqs=jnp.linspace(1, 50, 393), da_amp=4e6)
mock = simloader.load_mock_sim()


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

ratio = templates / templates.sel(freqs=10)
g = ratio.plot.line(row="kind", sharey=False)
g.axs[0, 0].set_yscale("log")
plt.show()
g

##--------------------------------------------------------------------##
# %%


# %%
