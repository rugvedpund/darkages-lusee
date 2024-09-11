import jax
import jax.numpy as jnp

import simflows.jax as simjax
import simflows.load as loader
import simflows.pca as simpca
import simflows.utils as simutils

# WARN: stupid jax doesn't like float64 by default
jax.config.update("jax_enable_x64", True)
import lusee
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xarray_einstats
