# %%

from dalusee.imports import *

# %%

DEFAULT_SEED = 42
DEFAULT_NITER = 10000
MASK_THRESHOLD = 12
MASK_SIZE = 50


def sim2pcaproj(sim) -> Tuple[xr.DataArray, xr.DataArray]:
    """Project simulation data onto PCA components."""
    delta, fg, da, cmb = sim.sel(kind=["delta", "fg", "da", "cmb"])
    delta_pca = simpca.get_pca(delta, "times", "freqs", other_dims=[])
    proj = simpca.get_pca_proj(
        fg, delta, da, cmb, delta_pca, "times", "freqs", other_dims=[]
    )
    return delta_pca, proj


def make_random_weights(
    motherseed: int = DEFAULT_SEED,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate random weights for DA and CMB."""
    rng = np.random.RandomState(motherseed)
    seed1, seed2 = rng.randint(0, 1000, 2)
    wda = make_w(seed1)
    wcmb = make_w(seed2)
    return wda, wcmb


def make_w(
    seed: int = DEFAULT_SEED,
    mask: jnp.ndarray = jnp.arange(MASK_SIZE) >= MASK_THRESHOLD,
) -> jnp.ndarray:
    """Generate a weight vector with a given seed and mask."""
    key = jax.random.PRNGKey(seed)
    w = jax.random.uniform(key, shape=(MASK_SIZE,)) - 0.5
    w = w / jnp.linalg.norm(w)
    wsigns = jnp.sign(w)
    w = w / wsigns[-1]  # enforce positive sign for first element
    w = jnp.where(mask, w, 0)  # replace mask=False with 0
    return w


def optimize(
    loss,
    wtensor: jnp.ndarray,
    deltafg: jnp.ndarray,
    signal: jnp.ndarray,
    niter: int = DEFAULT_NITER,
) -> Tuple[jnp.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Optimize the weight tensor using the given loss function."""
    learning_rate = 0.01
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(
            optax.exponential_decay(
                init_value=learning_rate,
                transition_steps=niter / 1,
                decay_rate=0.99,
            )
        ),
    )
    opt_state = optimizer.init(wtensor)

    @jax.jit
    def _step(w, _opt_state):
        loss_value, grads = jax.value_and_grad(loss)(w, deltafg, signal)
        updates, _opt_state = optimizer.update(grads, _opt_state)
        w = optax.apply_updates(w, updates)
        return w, _opt_state, loss_value

    iterations = np.zeros((niter,))
    loss_vals = np.zeros((niter,))
    wnorms = np.zeros((niter,))
    witers = np.zeros((niter, MASK_SIZE))

    for i in tqdm(range(niter), leave=False, desc="Optimizer"):
        witers[i] = wtensor
        wnorms[i] = jnp.linalg.norm(wtensor)
        wtensor, opt_state, loss_value = _step(wtensor, opt_state)
        iterations[i], loss_vals[i] = i, loss_value
    woptim = wtensor
    return woptim, iterations, loss_vals, wnorms, witers


def lossfn(w: jnp.ndarray, deltafg: jnp.ndarray, signal: jnp.ndarray) -> jnp.ndarray:
    """Calculate the loss function."""
    wnorm = jnp.linalg.norm(w)
    w = w / wnorm
    deltafg_tilde = jnp.einsum("f,tf->t", w, deltafg)
    signal_tilde = jnp.einsum("f,tf->t", w, signal).mean()
    return (jnp.var(deltafg_tilde) / signal_tilde**2) + (wnorm**2 - 1) ** 2


def optimize_wtensors(motherseed: int, niter: int) -> Tuple[xr.Dataset, xr.Dataset]:
    """Optimize weight tensors for DA and CMB."""
    sim = simloader.make_toysim(motherseed)
    _, jda, jcmb, jdelta = simloader.sim2jax(sim)
    wda, wcmb = make_random_weights(motherseed)
    woptim_da, iter_da, loss_vals_da, wnorms_da, witers_da = optimize(
        lossfn, wda, jdelta, jda, niter
    )
    woptim_cmb, iter_cmb, loss_vals_cmb, wnorms_cmb, witers_cmb = optimize(
        lossfn, wcmb, jdelta, jcmb, niter
    )
    assert np.allclose(iter_da, iter_cmb)
    iterations = xr.DataArray(iter_da, dims=("iter"))
    freqs = xr.DataArray(sim.freqs, dims=("freqs"))
    xrwda = xr.Dataset(
        {
            "wda0": xr.DataArray(wda, dims=("freqs")),
            "woptim": xr.DataArray(woptim_da, dims=("freqs")),
            "loss": xr.DataArray(loss_vals_da, dims=("iter")),
            "wnorm": xr.DataArray(wnorms_da, dims=("iter")),
            "witer": xr.DataArray(witers_da, dims=("iter", "freqs")),
        },
        coords={"freqs": freqs, "iter": iterations},
    )
    xrwcmb = xr.Dataset(
        {
            "wcmb0": xr.DataArray(wcmb, dims=("freqs")),
            "woptim": xr.DataArray(woptim_cmb, dims=("freqs")),
            "loss": xr.DataArray(loss_vals_cmb, dims=("iter")),
            "wnorm": xr.DataArray(wnorms_cmb, dims=("iter")),
            "witer": xr.DataArray(witers_cmb, dims=("iter", "freqs")),
        },
        coords={"freqs": freqs, "iter": iterations},
    )
    return xrwda, xrwcmb


def optimize_proj_wtensors(
    motherseed: int, niter: int
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Optimize projected weight tensors for DA and CMB."""
    sim = simloader.make_toysim(motherseed)
    delta_pca, proj = sim2pcaproj(sim)
    wda, wcmb = make_random_weights(motherseed)
    jpdelta, jpda, jpcmb = map(
        jnp.array,
        [
            proj.sel(kind="pdelta").data,
            proj.sel(kind="pda").data,
            proj.sel(kind="pcmb").data,
        ],
    )
    woptim_da, iter_da, loss_vals_da, wnorms_da, witers_da = optimize(
        lossfn, wda, jpdelta, jpda, niter
    )
    woptim_cmb, iter_cmb, loss_vals_cmb, wnorms_cmb, witers_cmb = optimize(
        lossfn, wcmb, jpdelta, jpcmb, niter
    )

    iterations = xr.DataArray(iter_da, dims=("iter"))
    freqs_eig = xr.DataArray(proj.freqs_eig, dims=("freqs_eig"))
    xrwda = xr.Dataset(
        {
            "wda0": xr.DataArray(wda, dims=("freqs_eig")),
            "woptim": xr.DataArray(woptim_da, dims=("freqs_eig")),
            "loss": xr.DataArray(loss_vals_da, dims=("iter")),
            "wnorm": xr.DataArray(wnorms_da, dims=("iter")),
            "witer": xr.DataArray(witers_da, dims=("iter", "freqs_eig")),
        },
        coords={"freqs_eig": freqs_eig, "iter": iterations},
    )
    xrwcmb = xr.Dataset(
        {
            "wcmb0": xr.DataArray(wcmb, dims=("freqs_eig")),
            "woptim": xr.DataArray(woptim_cmb, dims=("freqs_eig")),
            "loss": xr.DataArray(loss_vals_cmb, dims=("iter")),
            "wnorm": xr.DataArray(wnorms_cmb, dims=("iter")),
            "witer": xr.DataArray(witers_cmb, dims=("iter", "freqs_eig")),
        },
        coords={"freqs_eig": freqs_eig, "iter": iterations},
    )
    return xrwda, xrwcmb


def run_seeds(
    seeds: List[int], niter: int = DEFAULT_NITER
) -> Dict[int, Tuple[xr.Dataset, xr.Dataset]]:
    """Run optimization for multiple seeds."""
    results = dict()
    for seed in tqdm(seeds, desc="Seeds", leave=True):
        wpda, wpcmb = optimize_proj_wtensors(seed, niter)
        results[seed] = (wpda, wpcmb)
    return results


def plot_results(results: Dict[int, Tuple[xr.Dataset, xr.Dataset]], seeds: List[int]):
    """Plot the results of the optimization."""
    fig, ax = plt.subplots(2, 4, figsize=(12, 8))
    for seed in seeds:
        sim = simloader.make_toysim(seed)
        delta_pca, proj = sim2pcaproj(sim)
        wpda, wpcmb = results[seed]
        wpda.loss.plot(ax=ax[0, 0], yscale="log", alpha=0.7)
        wpda.wnorm.plot(ax=ax[0, 1])
        (axfreq,) = wpda.wda0.plot(ax=ax[0, 2])
        (axfeig,) = ax[0, 3].plot(
            xr.dot(wpda.wda0, delta_pca.U, dims="freqs_eig"), label="da"
        )
        ax[0, 3].set_xlabel("freqs")
        wpcmb.loss.plot(ax=ax[1, 0], yscale="log", alpha=0.7)
        wpcmb.wnorm.plot(ax=ax[1, 1])
        wpcmb.wcmb0.plot(ax=ax[1, 2])
        ax[1, 3].plot(xr.dot(wpcmb.wcmb0, delta_pca.U, dims="freqs_eig"), label="cmb")
        ax[1, 3].set_xlabel("freqs")
    plt.suptitle("optimize pca sim")
    plt.show()


# %%
