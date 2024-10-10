# Load the simulation data into tensors

import jax
import jax.numpy as jnp
import lusee
import numpy as np
import xarray as xr

# WARN: this dumb shit is jax because they dont like float64. it only works at startup
jax.config.update("jax_enable_x64", True)


##--------------------------------------------------------------------##


DEFAULT_SEED = 42
DEFAULT_NITER = 100000
DEFAULT_NTIMES = 650
DEFAULT_IDXS_MEAN = 2.5
DEFAULT_IDXS_SIGMA = 0.5
DEFAULT_AMP20MEAN = 1e4
DEFAULT_AMP20SIGMA = 1e4
MASK_THRESHOLD = 12
MASK_SIZE = 50


def random_normal(shape, seed=42, mean=0, sigma=1):
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, shape) * sigma + mean


def load_templates(
    freqs=np.linspace(1, 50),
    amp20: float = 1e5,
    idx: float = 2.54,
    T_cmb: float = 2.75,
    da_amp: float = 1.0,
):
    print("here")
    fg = fg_template(freqs, amp20, idx)
    da = da_template(freqs, da_amp)
    cmb = cmb_template(freqs, T_cmb)
    print("creating mock templates..")
    print("  ", f"{fg.shape=}", f"{da.shape=}", f"{cmb.shape=}")
    coords = {"freqs": freqs}  # Keep freqs as JAX array
    templates = xr.Dataset()
    templates["fg"] = xr.DataArray(np.array(fg), coords)  # Convert to NumPy array here
    templates["da"] = xr.DataArray(np.array(da), coords)  # Convert to NumPy array here
    templates["cmb"] = xr.DataArray(
        np.array(cmb), coords
    )  # Convert to NumPy array here
    return templates.to_dataarray(dim="kind")


def sim2jax(sim):
    delta, fg, da, cmb = sim.sel(kind=["delta", "fg", "da", "cmb"])
    jfg, jda, jcmb, jdelta = (
        jnp.array(fg),
        jnp.array(da),
        jnp.array(cmb),
        jnp.array(delta),
    )
    return jfg, jda, jcmb, jdelta


def make_toysim(motherseed: int = DEFAULT_SEED):
    rng = np.random.RandomState(motherseed)
    seed1, seed2 = rng.randint(0, 1000, 2)
    ntimes = DEFAULT_NTIMES
    idxs = random_normal(
        (ntimes, 1), seed=seed1, mean=DEFAULT_IDXS_MEAN, sigma=DEFAULT_IDXS_SIGMA
    )
    amp20MHz = random_normal(
        (ntimes, 1), seed=seed2, mean=DEFAULT_AMP20MEAN, sigma=DEFAULT_AMP20SIGMA
    )
    amp20MHz = jnp.abs(amp20MHz)
    sim = make_mock_sim(
        ntimes=ntimes,
        idxs=idxs,
        amp20MHz=amp20MHz,
        da_amp=1,
        freqs=jnp.linspace(1, 50, 50),
    )
    return sim


def make_mock_sim(
    freqs: np.ndarray = np.linspace(1, 50),
    fpivot: float = 20.0,
    ntimes: int = DEFAULT_NTIMES,
    amp20MHz: np.ndarray = None,
    idxs: np.ndarray = None,
    T_cmb: float = 2.75,
    da_amp: float = 1.0,
):
    # NOTE: creating jax arrays just to convert them back to numpy. to be fixed
    if idxs is None:
        print("  creating random idxs..")
        idxs = random_normal((ntimes, 1), seed=0, mean=2.5, sigma=0.1)
    if amp20MHz is None:
        print("  creating random amp30s..")
        amp20MHz = random_normal((ntimes, 1), seed=1, mean=1e5, sigma=1e5)
    # print("sigma = 0.0")
    # idxs = random_normal((ntimes, 1), seed=0, mean=2.5, sigma=0.0)
    amp20MHz = np.abs(amp20MHz)
    fg = fg_template(freqs, amp20MHz, idxs, fpivot)
    # print(f"{fg.shape=}")
    da = np.ones(ntimes)[:, None] * da_template(freqs, da_amp)
    cmb = np.ones(ntimes)[:, None] * cmb_template(freqs, T_cmb)
    # print("creating mock sims..")
    # print("  ", f"{fg.shape=}", f"{da.shape=}", f"{cmb.shape=}")
    mock = xr.Dataset()
    coords = {"times": np.arange(ntimes), "freqs": np.array(freqs)}
    mock["fg"] = xr.DataArray(fg, coords)
    mock["da"] = xr.DataArray(da, coords)
    mock["cmb"] = xr.DataArray(cmb, coords)
    mock["sum"] = mock["fg"] + mock["da"] + mock["cmb"]
    mock["delta"] = mock["sum"] - mock["sum"].mean("times")
    return mock.to_dataarray(dim="kind")


def da_template(freqs=jnp.linspace(1, 50), da_amp=1.0):
    """lusee T_DarkAges_Scaled template"""
    t21 = da_amp * lusee.monosky.T_DarkAges_Scaled(freqs)
    return jnp.array(t21)


def cmb_template(
    freqs: jnp.ndarray = jnp.linspace(1, 50), T_cmb: float = 2.75
) -> jnp.ndarray:
    """constant CMB template"""
    return T_cmb * jnp.ones_like(freqs)


def fg_template(
    freqs: jnp.ndarray = jnp.linspace(1, 50),
    amp20: float = 1e6,
    idx: float = 2.54,
    fpivot: float = 20.0,
) -> jnp.ndarray:
    """Power law function for the frequency domain"""
    return amp20 * (1 / (freqs / fpivot) ** idx)
