# %%

from simflows.imports import *

##--------------------------------------------------------------------##
# %%


class ToySim:
    def __init__(self, motherseed: int = 42):
        self.make_toysim(motherseed)

    def make_toysim(self, motherseed: int = 42):
        # print(f"creating ToySim object..")
        # print(f"  using {motherseed=}")
        rng = np.random.RandomState(motherseed)
        seed1, seed2 = rng.randint(0, 1000, 2)
        ntimes = 650
        idxs = simjax.random_normal((ntimes, 1), seed=seed1, mean=2.5, sigma=0.5)
        amp20MHz = simjax.random_normal((ntimes, 1), seed=seed2, mean=1e4, sigma=1e4)
        amp20MHz = jnp.abs(amp20MHz)
        sim = simloader.make_mock_sim(
            ntimes=ntimes, idxs=idxs, amp20MHz=amp20MHz, da_amp=1
        )
        fg, da, cmb, _, delta = sim
        self.sim = sim
        self.times = sim.times
        self.freqs = sim.freqs
        (
            self.jfg,
            self.jda,
            self.jcmb,
            self.jdelta,
        ) = (
            jnp.array(fg),
            jnp.array(da),
            jnp.array(cmb),
            jnp.array(delta),
        )
        # jda = jda.at[-1].set(1e-4)
        return self.jfg, self.jda, self.jcmb, self.jdelta, self.freqs


class WeightsTensor:
    def __init__(self, motherseed: int = 42):
        # print("creating WeightsTensor object..")
        # print(f"  using jax PRNG {motherseed=}")
        rng = np.random.RandomState(motherseed)
        seed1, seed2 = rng.randint(0, 1000, 2)
        self.wda = self.make_random(seed1)
        self.wcmb = self.make_random(seed2)

    def make_random(self, seed: int = 42):
        key = jax.random.PRNGKey(seed)
        w = jax.random.uniform(key, shape=(50)) - 0.5
        w = w / jnp.linalg.norm(w)
        wsigns = jnp.sign(w)
        w = w / wsigns[20]  # enforce positive sign for first element
        return w


def lossfn(w, deltafg, signal):
    wnorm = jnp.linalg.norm(w)  # frobenius norm sqrt(sum(w**2))
    w = w / wnorm
    deltafg_tilde = jnp.einsum("f,tf->t", w, deltafg)
    signal_tilde = jnp.einsum("f,tf->t", w, signal)
    signal_tilde = signal_tilde.mean()
    return (jnp.var(deltafg_tilde) / signal_tilde**2) + (wnorm**2 - 1) ** 2


def jax_optimize(lossfn, wtensor, deltafg, signal, learning_rate, niter):
    optimizer = optax.adam(learning_rate)
    schedule = optax.exponential_decay(
        init_value=learning_rate, transition_steps=niter / 1, decay_rate=0.99
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = optimizer.init(wtensor)

    @jax.jit
    def _step(w, _opt_state):
        loss_value, grads = jax.value_and_grad(lossfn)(w, deltafg, signal)
        updates, _opt_state = optimizer.update(grads, _opt_state)
        w = optax.apply_updates(w, updates)
        return w, _opt_state, loss_value

    iterations = np.zeros((niter))
    loss_vals = np.zeros((niter))
    wnorms = np.zeros((niter))
    witers = np.zeros((niter, 50))

    for i in tqdm(range(niter), leave=False, desc="optim"):
        witers[i] = wtensor.copy()
        wnorms[i] = jnp.linalg.norm(wtensor)
        wtensor, opt_state, loss_value = _step(wtensor, opt_state)
        iterations[i], loss_vals[i] = i, loss_value
    woptim = wtensor
    return woptim, iterations, loss_vals, wnorms, witers


def run_optimizers(seed, learning_rate, niter):
    # print(" seed:", seed)
    toysim = ToySim(seed)
    wtensor = WeightsTensor(seed)
    woptim_da, iterations_da, loss_vals_da, wnorms_da, witers_da = jax_optimize(
        lossfn, wtensor.wda, toysim.jdelta, toysim.jda, learning_rate, niter
    )
    woptim_cmb, iterations_cmb, loss_vals_cmb, wnorms_cmb, witers_cmb = jax_optimize(
        lossfn, wtensor.wcmb, toysim.jdelta, toysim.jcmb, learning_rate, niter
    )
    optimized = dict()
    optimized["da"] = (woptim_da, iterations_da, loss_vals_da, wnorms_da, witers_da)
    optimized["cmb"] = (
        woptim_cmb,
        iterations_cmb,
        loss_vals_cmb,
        wnorms_cmb,
        witers_cmb,
    )
    return optimized


def run_seeds(nseeds, learning_rate, niter):
    results = dict()
    results["nseeds"] = nseeds
    results["learning_rate"] = learning_rate
    results["niter"] = niter
    for seed in tqdm(range(nseeds), leave=False, desc="seeds"):
        # print(f"seed: {seed}")
        results[f"{seed}"] = run_optimizers(seed, learning_rate, niter)
    return results


def plot_seeds(freqs, seeds_dict):
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    for isig, signal in enumerate(["da", "cmb"]):
        for i in range(nseeds):
            woptim, iterations, loss_vals, wnorms, witers = seeds_dict[f"{i}"][signal]
            ax[isig, 0].plot(iterations, wnorms, label=f"seed {i}", alpha=0.5)
            ax[isig, 1].semilogy(iterations, loss_vals, label=f"seed {i}", alpha=0.5)
            ax[isig, 2].plot(freqs, woptim, label=f"seed {i}", color=f"C{i}")
        ax[isig, 0].set_title(f"{signal} wnorms")
        ax[isig, 0].set_xlabel("iterations")
        ax[isig, 1].set_title(f"{signal} loss")
        ax[isig, 1].set_xlabel("iterations")
        ax[isig, 2].set_title(f"{signal} woptim")
        ax[isig, 2].set_xlabel("frequency")
    plt.tight_layout()


# %%

seed = 0
toysim = ToySim(seed)
wtensor = WeightsTensor(seed)
lossfn = lossfn

nseeds = 10
learning_rate = 0.01
niter = 100000
allseeds = run_seeds(nseeds, learning_rate, niter)

plot_seeds(toysim.freqs, allseeds)
plt.show()


# %%

plot_seeds(toysim.freqs, allseeds)
plt.show()

# %%


def get_witer(allseeds, signal, seed, iter):
    return allseeds[f"{seed}"][signal][4][iter - 1]


def get_wxjdelta(allseeds, signal, seed, iter):
    return jnp.einsum("f,tf->t", get_witer(allseeds, signal, seed, iter), toysim.jdelta)


# %%

wdaxjdelta0 = get_wxjdelta(allseeds, "da", seed=0, iter=0)
wcmbxjdelta0 = get_wxjdelta(allseeds, "cmb", seed=0, iter=0)

iter = -1
wdaxjdelta = get_wxjdelta(allseeds, "da", 0, iter)
wcmbxjdelta = get_wxjdelta(allseeds, "cmb", 0, iter)

plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.hist(wdaxjdelta0, bins=100, label="wda init", density=True)
plt.hist(wdaxjdelta, bins=100, label="wda final", density=True)
plt.title("wda weighted deltafg")
plt.legend()
plt.subplot(122)
plt.hist(wcmbxjdelta0, bins=100, label="wcmb init", density=True)
plt.hist(wcmbxjdelta, bins=100, label="wcmb final", density=True)
plt.legend()
plt.title("wcmb weighted deltafg")
plt.show()


# %%

delta, fg, da, cmb = toysim.sim.sel(kind=["delta", "fg", "da", "cmb"])
delta_pca = simpca.get_pca(delta, "times", "freqs", other_dims=[])
proj = simpca.get_pca_proj(fg, da, cmb, delta_pca, "times", "freqs", [])
loss3 = (
    proj.sel(kind="pfg").var("times")
    / proj.sel(kind="pda").mean("times").sum("freqs_eig") ** 2
)
eigmodes = loss3.freqs_eig.astype(int).astype(str).data
plt.scatter(eigmodes, loss3)
plt.yscale("log")
plt.xlabel("eigenmodes")
plt.show()

# %%
##--------------------------------------------------------------------##
# test multiple seeds


# def animate_seeds(freqs, wdict):
#     fig, ax = plt.subplots(2, 3, figsize=(12, 6))
#     frames = range(0, wdict["niter"], 100)

# def _update(frame):
#     for isig, signal in enumerate(["da", "cmb"]):
#         lnorms = ax[isig, 0].axvline(0)
#         lloss = ax[isig, 1].axvline(0)
#         (lweights,) = ax[isig, 2].plot([], [])
#         for i in range(wdict["nseeds"]):
#             woptim, iterations, loss_vals, wnorms, witers = wdict[f"{signal}_{i}"]
#             # wnorms, loss, witers for da and cmb
#             ax[isig, 0].plot(iterations, wnorms, label=f"seed {i}")
#             ax[isig, 1].plot(iterations, loss_vals, label=f"seed {i}")
#             ax[isig, 2].plot(
#                 freqs, woptim, label=f"seed {i}", color=f"C{i}", linestyle="--"
#             )
#             lnorms.set_xdata([iterations[frame], iterations[frame]])
#             lloss.set_xdata([iterations[frame], iterations[frame]])
#             lweights.set_ydata(witers[frame])
#             lims = max(abs(witers[frame].min()), abs(witers[frame].max()))
#             ax[isig, 2].set_ylim(-lims, lims)
#             ax[isig, 2].set_title(f"iteration {frame}")
#     return lnorms, lloss, lweights

# ani = animation.FuncAnimation(fig, _update, frames=frames, interval=10)
# plt.show()


# freqs = np.linspace(1, 50, 50)
# wdict = run_seeds(10, 0.01, 50000)
# plot_seeds(freqs, wdict)


# %%
##--------------------------------------------------------------------##
# test single seed, this animation works

motherseed = 0
learning_rate, niter = 0.01, 100000
jsig = "cmb"
woptim, iterations, loss_vals, wnorms, witers = run_optimizers(
    jsig, motherseed, learning_rate, niter
)

# plot the loss and wnorm
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(iterations, loss_vals)
plt.yscale("log")
plt.title("jax loss")
plt.subplot(122)
plt.plot(iterations, wnorms)
plt.title("wnorm")
plt.show()

frames = range(0, niter, 100)
fig, ax = plt.subplots()
(line,) = ax.plot(freqs, witers[0], color="blue")
# Set plot limits
ax.set_xlim(freqs.min(), freqs.max())
ax.set_xlabel("Frequency")
ax.set_ylabel("Weight")


# Update function to update the plot for each frame
def update(frame):
    line.set_ydata(witers[frame])
    lims = max(abs(witers[frame].min()), abs(witers[frame].max()))
    ax.set_ylim(-lims, lims)
    ax.set_title(f"{jsig}: iteration {frame}")
    # ax.set_text(0.5, 0.9, f"iteration {frame}", transform=ax.transAxes)
    return (line,)


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=10)
writergif = animation.PillowWriter(fps=60)
# ani.save(f"{jsig}_optimization_progress.gif", writer=writergif)
plt.show()
