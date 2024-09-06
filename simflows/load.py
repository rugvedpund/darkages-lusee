# Load the simulation data into tensors

import os

import jax
import jax.numpy as jnp
import lusee
import numpy as np
import xarray as xr

import simflows.utils as simutils

# WARN: this dumb shit is jax because they dont like float64. it only works at startup
jax.config.update("jax_enable_x64", True)


# path_here = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.join(path_here, "../")


##--------------------------------------------------------------------##
# %%


def load_templates(
    freqs: jnp.ndarray = jnp.linspace(1, 50),
    ntimes: int = 650,
    sigma: float = 1e-3,
    amp30: jnp.float64 = 2e4,
    idx: jnp.float64 = 2.54,
    T_cmb: jnp.float32 = 2.75,
    da_amp: jnp.float32 = 1.0,
):
    # NOTE: creating jax arrays just to convert them back to numpy. to be fixed
    fg = fg_template(freqs, amp30, idx)
    da = da_template(freqs, da_amp)
    cmb = cmb_template(freqs, T_cmb)
    print("creating mock templates..")
    print("  ", f"{fg.shape=}", f"{da.shape=}", f"{cmb.shape=}")
    coords = {"freqs": np.array(freqs)}
    templates = xr.Dataset()
    templates["fg"] = xr.DataArray(fg, coords)
    templates["da"] = xr.DataArray(da, coords)
    templates["cmb"] = xr.DataArray(cmb, coords)
    return templates.to_dataarray(dim="kind")


def load_mock_sim(
    freqs: jnp.ndarray = jnp.linspace(1, 50),
    ntimes: int = 650,
    sigma: float = 1e-3,
    amp30: jnp.float64 = 1e5,
    idx: jnp.float64 = 2.54,
    T_cmb: jnp.float32 = 2.75,
):
    # NOTE: creating jax arrays just to convert them back to numpy. to be fixed
    fg = fg_template(freqs, amp30, idx)[:, None] * jnp.ones(ntimes)
    da = da_template(freqs)[:, None] * jnp.ones(ntimes)
    cmb = cmb_template(freqs, T_cmb)[:, None] * jnp.ones(ntimes)
    noise = gaussian_noise(ntimes, sigma)
    print("creating mock sims..")
    print("  ", f"{fg.shape=}", f"{da.shape=}", f"{cmb.shape=}", f"{noise.shape=}")

    mock = xr.Dataset()
    coords = {"freqs": np.array(freqs), "times": np.arange(ntimes)}
    mock["fg"] = xr.DataArray(fg, coords)
    mock["da"] = xr.DataArray(da, coords)
    mock["cmb"] = xr.DataArray(cmb, coords)
    mock["noise"] = xr.DataArray(noise, coords)
    return mock.to_dataarray(dim="kind")


# %%


def gaussian_noise(
    ntimes: int = 650, sigma: float = 1e-3, freqs: jnp.ndarray = jnp.linspace(1, 50)
) -> jnp.ndarray:
    """Gaussian noise"""
    key = jax.random.PRNGKey(42)
    nfreqs = len(freqs)
    return sigma * jax.random.normal(key, (nfreqs, ntimes))


def da_template(
    freqs: jnp.ndarray = jnp.linspace(1, 50), da_amp: jnp.float32 = 1.0
) -> jnp.ndarray:
    """lusee T_DarkAges_Scaled template"""
    t21 = da_amp * lusee.monosky.T_DarkAges_Scaled(freqs)
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
