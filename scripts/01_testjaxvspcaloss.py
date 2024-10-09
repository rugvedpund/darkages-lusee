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


# pickle.dump(results, open("./data/outputs/results.pkl", "wb"))

# results = pickle.load(open("./data/outputs/results0.pkl", "rb"))
results = pickle.load(open("./data/outputs/results.pkl", "rb"))

##--------------------------------------------------------------------##
# plots

# %%
# maybe the problem is with pca loss computation
# proj fg index must be contracted with eig index
seed = 0
sim = simloader.make_toysim(seed)
delta_pca, proj = sim2pcaproj(sim)
wpda, wpcmb = results[seed]
jfg, jpda, jpcmb, jpdelta = simloader.sim2jax(sim)
pca_loss_da = sim.sel(kind="delta").var("times") / sim.sel(kind="da").mean("times") ** 2
pca_loss_cmb = (
    sim.sel(kind="delta").var("times") / sim.sel(kind="cmb").mean("times") ** 2
)

wpda_loss = wpda.loss.isel(iter=-1) - (wpda.wnorm.isel(iter=-1) - 1) ** 2
wpcmb_loss = wpcmb.loss.isel(iter=-1) - (wpcmb.wnorm.isel(iter=-1) - 1) ** 2

plt.figure(figsize=(10, 5))
plt.subplot(121)
sim.sel(kind="fg").plot(
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
# %%
# plot dot product of woptims with pca modes
# delta_pca.U.isel
wpda.woptim.plot()
plt.show()


##--------------------------------------------------------------------##

# %%
# fig 3: fixed seed, jax loss vs loss per pca mode

seed = 0
sim = simloader.make_toysim(seed)
delta_pca, proj = sim2pcaproj(sim)
wpda, wpcmb = results[seed]
jfg, jpda, jpcmb, jpdelta = simloader.sim2jax(sim)
pca_loss_da = (
    proj.sel(kind="pdelta").var("times") / proj.sel(kind="mean pda").mean("times") ** 2
)
pca_loss_cmb = (
    proj.sel(kind="pdelta").var("times") / proj.sel(kind="mean pcmb").mean("times") ** 2
)
wpda_loss = wpda.loss.isel(iter=-1) - (wpda.wnorm.isel(iter=-1) - 1) ** 2
wpcmb_loss = wpcmb.loss.isel(iter=-1) - (wpcmb.wnorm.isel(iter=-1) - 1) ** 2

# %%
# fig: fg dist, jax loss vs loss per pca mode

plt.figure(figsize=(10, 5))
plt.subplot(121)
sim.sel(kind="fg").plot(
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
# %%
# fig 1: plot all seeds with pca before optimization

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
# fig 2: plot all seeds, with and without pca before optimization

fig, ax = plt.subplots(2, 5, figsize=(15, 10))
for seed in seeds:
    sim = simloader.make_toysim(seed)
    delta_pca, proj = sim2pcaproj(sim)
    wpda, wpcmb = results[seed]
    pca_loss_da = proj.sel(kind="pdelta").var("times") / proj.sel(kind="mean pda") ** 2
    pca_loss_da_num = proj.sel(kind="pdelta").var("times")
    pca_loss_da_den = proj.sel(kind="mean pda") ** 2
    pca_loss_cmb = (
        proj.sel(kind="pdelta").var("times") / proj.sel(kind="mean pcmb") ** 2
    )
    pca_loss_cmb_num = proj.sel(kind="pdelta").var("times")
    pca_loss_cmb_den = proj.sel(kind="mean pcmb") ** 2
    wpda_loss = wpda.loss.isel(iter=slice(-100, None)).mean("iter")
    wpcmb_loss = wpcmb.loss.isel(iter=slice(-100, None)).mean("iter")

    ax[0, 0].plot(xr.dot(wpda.woptim, delta_pca.U, dims="freqs_eig"), color=f"C{seed}")
    ax[0, 0].set_xlabel("freqs")
    ax[0, 0].set_title("da optimize pca")
    ax[0, 1].set_title("da optimize w/o pca")
    pca_loss_da.plot.scatter(yscale="log", ax=ax[0, 2], color="C0")
    ax[0, 2].axhline(wpda_loss, color=f"C{seed}", linestyle="--")
    ax[0, 2].set_title("jax loss vs naive pca")
    # ax[0, 3].plot(pca_loss_da_num.mean("times"), color="C0")
    pca_loss_da_num.plot(ax=ax[0, 3], color="C0", x="freqs_eig")
    ax[0, 3].set_yscale("log")
    pca_loss_da_den.mean("times").plot(ax=ax[0, 4], color="C0", x="freqs_eig")
    ax[0, 4].set_yscale("log")

    ax[1, 0].plot(
        xr.dot(wpcmb.woptim, delta_pca.U, dims="freqs_eig"),
        color=f"C{seed}",
    )
    ax[1, 0].set_xlabel("freqs")
    ax[1, 0].set_title("cmb optimize pca")
    ax[1, 1].set_title("cmb optimize w/o pca")
    pca_loss_cmb.plot.scatter(yscale="log", ax=ax[1, 2], color="C1")
    ax[1, 2].axhline(wpcmb_loss, color=f"C{seed}", linestyle="--")
    ax[1, 2].set_title("jax loss vs loss per pca mode")
    pca_loss_cmb_num.plot(ax=ax[1, 3], color="C0", x="freqs_eig")
    ax[1, 3].set_yscale("log")
    pca_loss_cmb_den.mean("times").plot(ax=ax[1, 4], color="C0", x="freqs_eig")
    ax[1, 4].set_yscale("log")
    # ax[1, 3].plot(pca_loss_cmb_num.mean("times"), color="C1")
    # ax[1, 3].set_yscale("log")
    # ax[1, 4].plot(pca_loss_cmb_den.mean("times"), color="C0")
    # ax[1, 4].set_yscale("log")

ax[0, 2].plot([], [], color="k", label="jax loss w/o pca")
ax[0, 2].plot([], [], color="k", linestyle="--", label="loss w pca")
ax[0, 2].legend()
ax[1, 2].plot([], [], color="k", label="jax loss w/o pca")
ax[1, 2].plot([], [], color="k", linestyle="--", label="loss w pca")
ax[1, 2].legend()
plt.show()

# %%
# fig 4: plot w and wo pca on one plot
plt.plot(xr.dot(wpda.woptim, delta_pca.U, dims="freqs_eig"), label="da")
plt.plot(xr.dot(wpcmb.woptim, delta_pca.U, dims="freqs_eig"), label="cmb")
wda.woptim.plot(color="C0", ls="--", alpha=0.7)
wcmb.woptim.plot(color="C1", ls="--", alpha=0.7)
plt.plot([], [], color="k", ls="--", label="w/o pca")
plt.legend()
plt.show()

# %%
# fig 5: plot likelihoods for all seeds


def likelihood(w, meanulsa, meansig, amp):
    monopole = meanulsa + (1 - amp)[:, None] * meansig
    return jnp.einsum("f,af->a", w, monopole)


maxamp = 1e10
amp = jnp.linspace(-maxamp, maxamp, 100)

seed = 0
sim = simloader.make_toysim(seed)
delta_pca, proj = sim2pcaproj(sim)
wda, wcmb, wpda, wpcmb = results[seed]
jfg, jda, jcmb, jdelta = simloader.sim2jax(sim)
plt.plot(amp, likelihood(wda.woptim.data, jfg.mean(axis=0), jda.mean(axis=0), amp))
# plt.yscale("symlog")
plt.show()

# %%

# %%

# niter = 100000
# seed = 0
# sim = make_toysim()
# jfg, jpda, jpcmb, jpdelta = sim_to_jax(sim)
# delta_pca, proj = doPCA(sim)
# wda0, wcmb0 = make_random_weights(seed)
# wda, wcmb = optimize_wtensors(seed, niter)
# wpda, wpcmb = optimize_proj_wtensors(seed, niter)

# %%

# %%

# %%
