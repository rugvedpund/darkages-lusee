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

templates = loader.load_templates()
mock = loader.load_mock_sim()

##--------------------------------------------------------------------##
# %%

# templates for ulsa,da,cmb
templates.plot.line(row="kind", sharey=False)
plt.show()

##--------------------------------------------------------------------##
# %%


# %%
