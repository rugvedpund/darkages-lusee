# Load the simulation data into tensors

import jax
import jax.numpy as jnp
import lusee
import numpy as np
import xarray as xr

import simflows.jax as simjax

# WARN: this dumb shit is jax because they dont like float64. it only works at startup
jax.config.update("jax_enable_x64", True)


# path_here = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.join(path_here, "../")


##--------------------------------------------------------------------##
# %%


def load_templates(
    freqs: jnp.ndarray = jnp.linspace(1, 50),
    amp20: jnp.float64 = 1e5,
    idx: jnp.float64 = 2.54,
    T_cmb: jnp.float32 = 2.75,
    da_amp: jnp.float32 = 1.0,
):
    # FIX: creating jax arrays just to convert them back to numpy. to be fixed
    fg = fg_template(freqs, amp20, idx)
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


# %%


def load_mock_sim(
    freqs: jnp.ndarray = jnp.linspace(1, 50),
    fpivot: jnp.float64 = 20.0,
    ntimes: int = 650,
    amp20MHz: jnp.ndarray = None,
    idxs: jnp.ndarray = None,
    T_cmb: jnp.float32 = 2.75,
    da_amp: jnp.float32 = 1.0,
):
    # NOTE: creating jax arrays just to convert them back to numpy. to be fixed
    if idxs is None:
        print("  creating random idxs..")
        idxs = simjax.random_normal((ntimes, 1), seed=0, mean=2.5, sigma=0.1)
    if amp20MHz is None:
        print("  creating random amp30s..")
        amp20MHz = simjax.random_normal((ntimes, 1), seed=1, mean=1e5, sigma=1e5)

    amp20MHz = jnp.abs(amp20MHz)
    fg = fg_template(freqs, amp20MHz, idxs, fpivot)
    print(f"{fg.shape=}")
    da = jnp.ones(ntimes)[:, None] * da_template(freqs, da_amp)
    cmb = jnp.ones(ntimes)[:, None] * cmb_template(freqs, T_cmb)
    print("creating mock sims..")
    print("  ", f"{fg.shape=}", f"{da.shape=}", f"{cmb.shape=}")

    mock = xr.Dataset()
    coords = {"times": np.arange(ntimes), "freqs": np.array(freqs)}
    mock["fg"] = xr.DataArray(fg, coords)
    mock["da"] = xr.DataArray(da, coords)
    mock["cmb"] = xr.DataArray(cmb, coords)
    mock["sum"] = mock["fg"] + mock["da"] + mock["cmb"]
    mock["delta"] = mock["sum"] - mock["sum"].mean("times")

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
    amp20: jnp.float64 = 1e6,
    idx: jnp.float64 = 2.54,
    fpivot: jnp.float64 = 20.0,
) -> jnp.ndarray:
    """Power law function for the frequency domain"""
    return amp20 * (1 / (freqs / fpivot) ** idx)


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
