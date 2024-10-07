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
mock = simloader.make_mock_sim()


g = templates.plot.line(row="kind", sharey=False)
g.axs[0, 0].set_yscale("log")
plt.show()
g

##--------------------------------------------------------------------##
# %%

sum = templates.sum("kind")
sum /= sum.sel(freqs=30)
dsumdnu = xr.DataArray(np.gradient(sum), dims=sum.dims, coords=sum.coords)

sum.plot(yscale="log")
dsumdnu.plot(yscale="symlog")
# plt.xlim(10,50)
plt.show()

# %%
