# %%
from dalusee.imports import *

# %%


def sim2pcaproj(sim):
    delta, fg, da, cmb = sim.sel(kind=["delta", "fg", "da", "cmb"])
    delta_pca = simpca.get_pca(delta, "times", "freqs", other_dims=[])
    proj = simpca.get_pca_proj(
        fg, delta, da, cmb, delta_pca, "times", "freqs", other_dims=[]
    )
    return delta_pca, proj


def make_random_weights(motherseed: int = 42):
    rng = np.random.RandomState(motherseed)
    seed1, seed2 = rng.randint(0, 1000, 2)
    wda = make_w(seed1)
    wcmb = make_w(seed2)
    return wda, wcmb


def make_w(seed: int = 42, mask=jnp.arange(50) >= 12):
    key = jax.random.PRNGKey(seed)
    w = jax.random.uniform(key, shape=(50)) - 0.5
    w = w / jnp.linalg.norm(w)
    wsigns = jnp.sign(w)
    w = w / wsigns[-1]  # enforce positive sign for first element
    w = jnp.where(mask, w, 0)  # replace mask=False with 0
    return w


def optimize(loss, wtensor, deltafg, signal, niter):
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

    iterations = np.zeros((niter))
    loss_vals = np.zeros((niter))
    wnorms = np.zeros((niter))
    witers = np.zeros((niter, 50))

    for i in tqdm(range(niter), leave=False, desc="Optimizer"):
        witers[i] = wtensor
        wnorms[i] = jnp.linalg.norm(wtensor)
        wtensor, opt_state, loss_value = _step(wtensor, opt_state)
        iterations[i], loss_vals[i] = i, loss_value
    woptim = wtensor
    return woptim, iterations, loss_vals, wnorms, witers


def lossfn(w, deltafg, signal):
    wnorm = jnp.linalg.norm(w)  # frobenius norm sqrt(sum(w**2))
    w = w / wnorm
    deltafg_tilde = jnp.einsum("f,tf->t", w, deltafg)
    signal_tilde = jnp.einsum("f,tf->t", w, signal)
    signal_tilde = signal_tilde.mean()
    return (jnp.var(deltafg_tilde) / signal_tilde**2) + (wnorm**2 - 1) ** 2


def optimize_wtensors(motherseed, niter):
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


def optimize_proj_wtensors(motherseed, niter):
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


def run_seeds(seeds: list[int], niter):
    results = dict()
    for seed in tqdm(seeds, desc="Seeds", leave=True):
        # wda, wcmb = optimize_wtensors(seed, niter)
        wpda, wpcmb = optimize_proj_wtensors(seed, niter)
        # results[seed] = (wda, wcmb, wpda, wpcmb)
        results[seed] = (wpda, wpcmb)
    return results


# %%

niter = 50000
seeds = np.arange(10)
results = run_seeds(seeds, niter)

# %%


# pickle.dump(results, open("./data/outputs/results0.pkl", "wb"))

# results = pickle.load(open("./data/outputs/results0.pkl", "rb"))
results = pickle.load(open("./data/outputs/results.pkl", "rb"))

##--------------------------------------------------------------------##
# plots

# %%

seed = 0
sim = simloader.make_toysim(seed)
fg, da, cmb, delta = sim.sel(kind=["fg", "da", "cmb", "delta"])
da_snr_pca = da.mean("times") ** 2 / delta.var("times")
cmb_snr_pca = cmb.mean("times") ** 2 / delta.var("times")

delta_pca, proj = sim2pcaproj(sim)
pfg, pda, pcmb, pdelta = proj.sel(kind=["pfg", "pda", "pcmb", "pdelta"])
mpfg, mpda, mpcmb = proj.sel(kind=["mean pfg", "mean pda", "mean pcmb"]).mean("times")

aa = np.linspace(-1e10, 1e10, 100)
amp = xr.DataArray(aa, dims=("amp"), coords={"amp": aa})

wpda, wpcmb = results[seed]
da_monopole = mpfg + (1 - amp) * mpda
cmb_monopole = mpfg + (1 - amp) * mpcmb

da_snr = xr.dot(wpda.woptim, da_monopole, dims="freqs_eig")
cmb_snr = xr.dot(wpcmb.woptim, cmb_monopole, dims="freqs_eig")

# %%

da_snr.plot(label="da")
cmb_snr.plot(label="cmb")
plt.legend()
plt.yscale("symlog")
plt.show()

# %%

wda_pdelta = xr.dot(wpda.woptim, pdelta, dims="freqs_eig")
wcmb_pdelta = xr.dot(wpcmb.woptim, pdelta, dims="freqs_eig")

wda_pdelta.plot.hist(label="da", xscale="symlog", yscale="log")
plt.show()

wcmb_pdelta.plot.hist(label="cmb", xscale="symlog", yscale="log")
plt.show()


# %%


# %%

##--------------------------------------------------------------------##
# %%
# fig: fg dist, jax loss vs loss per pca mode
# maybe the problem is with pca loss computation
# proj fg index must be contracted with eig index
seed = 0
sim = simloader.make_toysim(seed)
delta_pca, proj = sim2pcaproj(sim)
wpda, wpcmb = results[seed]
fg, da, cmb, delta = sim.sel(kind=["fg", "da", "cmb", "delta"])
pfg, pda, pcmb, pdelta = proj.sel(kind=["pfg", "pda", "pcmb", "pdelta"])
pca_loss_da = delta.var("times") / da.mean("times") ** 2
pca_loss_cmb = delta.var("times") / cmb.mean("times") ** 2
# pca_loss_da = pdelta.var("times") / pda.mean("times") ** 2
# pca_loss_cmb = pdelta.var("times") / pcmb.mean("times") ** 2
wpda_loss = wpda.loss.isel(iter=-1) - (wpda.wnorm.isel(iter=-1) - 1) ** 2
wpcmb_loss = wpcmb.loss.isel(iter=-1) - (wpcmb.wnorm.isel(iter=-1) - 1) ** 2


plt.figure(figsize=(10, 5))
plt.subplot(121)
fg.plot(
    x="freqs",
    hue="times",
    add_legend=False,
    yscale="log",
    color="grey",
    alpha=0.1,
)
plt.subplot(122)
pca_loss_da.plot.scatter(label="da", yscale="log")
pca_loss_cmb.plot.scatter(label="cmb", yscale="log")
plt.axhline(wpda_loss, color="C0", label="wda", linestyle="--")
plt.axhline(wpcmb_loss, color="C1", label="wcmb", linestyle="--")
plt.legend()
plt.title("jax loss vs loss per pca mode")
plt.show()


# %%
# plot dot product of woptims with pca modes
# delta_pca.U.isel
wpda.woptim.plot()
plt.show()


# %%
##--------------------------------------------------------------------##
# fig 3: fixed seed, jax loss vs loss per pca mode
