# Load the simulation data into tensors

import os

import jax.numpy as jnp
import lusee
import numpy as np
import xarray as xr

import simflows.utils as simutils

path_here = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path_here, "../")


##--------------------------------------------------------------------##
# %%

"""
1. need mock data for ulsa, da, cmb
    - ulsa: power law freqs x gaussian noise times
    - da,cmb: template freqs x no noise times
2. create named tensor class?
"""


def gaussian_noise(times: int = 650, sigma: float = 1e-3) -> jnp.ndarray:
    """Gaussian noise"""
    key = jax.random.PRNGKey(42)
    return sigma * jax.random.normal(key, (times,))


def da_template(freqs: jnp.ndarray = jnp.linspace(1, 50)) -> jnp.ndarray:
    """lusee T_DarkAges_Scaled template"""
    t21 = lusee.monosky.T_DarkAges_Scaled(freqs)
    return jnp.array(t21)


def cmb_template(
    freqs: jnp.ndarray = jnp.linspace(1, 50), T_cmb: jnp.float32 = 2.75
) -> jnp.ndarray:
    """constant CMB template"""
    return T_cmb * jnp.ones_like(freqs)


def fg_template(
    freqs: jnp.ndarray = jnp.linspace(1, 50),
    amp30: jnp.float64 = 1e5,
    idx: jnp.float64 = 2.54,
) -> jnp.ndarray:
    """Power law function for the frequency domain"""
    return amp30 * (1 / (freqs / 30) ** idx)


# %%

# # NOTE: argparser needs something like: python simtensors.py --config configs/a{0..80..10}_dt3600.yaml
# #
# # Load the lusee ulsa sim tensor
# print("Loading ULSA sim tensor..")
# parser = simutils.create_parser()
# args = parser.parse_args()
# sim = simutils.SimTensor().from_args(args)
# print("  saving as netcdf..")
# sim.to_netcdf("data/netcdf/sim.nc")

# # WARN: when loading netcdf tensor, ensure string coordniates are loaded as strings
#
# print("loading netcdf tensors..")
# sim = xr.open_dataset("../data/netcdf/sim.nc", chunks={"times": 650})
# sim["combs"] = sim["combs"].astype(str)
# sim = sim.sel(combs=simutils.comblist)
# with open("../data/netcdf/sim_hosvd.nc", "rb") as f:
#     eigs = xr.open_dataset(f)[["eig:combs", "eig:angles"]]
# eigs["combs"] = eigs["combs"].astype(str)
# eigs = eigs.sel(combs=simutils.comblist)
#
# psim = xr.Dataset()
# psim["ulsa"] = sim["ulsa"] @ eigs["eig:combs"] @ eigs["eig:angles"]
# psim["da"] = sim["da"] @ eigs["eig:combs"] @ eigs["eig:angles"]
# psim["cmb"] = sim["cmb"] @ eigs["eig:combs"] @ eigs["eig:angles"]
# data = psim["ulsa"] - psim["ulsa"].mean("times")
