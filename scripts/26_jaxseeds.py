# %%

from simflows.imports import *

##--------------------------------------------------------------------##
# %%

motherseed = 1
np.random.seed(motherseed)
seed1, seed2, seed3 = np.random.randint(0, 1000, 3)
print(f"  {seed1=}", f"{seed2=}", f"{seed3=}")


def load_toysims(seed1, seed2):
    # load the mock simulation
    ntimes = 650
    idxs = simjax.random_normal((ntimes, 1), seed=seed1, mean=2.5, sigma=0.5)
    amp20MHz = simjax.random_normal((ntimes, 1), seed=seed2, mean=1e4, sigma=1e4)
    amp20MHz = jnp.abs(amp20MHz)
    mock = simloader.make_mock_sim(idxs=idxs, amp20MHz=amp20MHz, da_amp=1)
    fg, da, cmb, _, delta = mock
    freqs = mock.freqs
    print("converting to jax arrays..")
    (
        jfg,
        jda,
        jcmb,
        jdelta,
    ) = (
        jnp.array(fg),
        jnp.array(da),
        jnp.array(cmb),
        jnp.array(delta),
    )
    # jda = jda.at[-1].set(1e-4)
    return jfg, jda, jcmb, jdelta, freqs


def get_wtensor(seed3):
    print("init random wtensor..")
    key = jax.random.PRNGKey(seed3)
    wtensor = jax.random.uniform(key, shape=(50)) - 0.5
    wtensor = wtensor / jnp.linalg.norm(wtensor)
    return wtensor


def lossfn(w, deltafg, signal):
    wnorm = jnp.linalg.norm(w)  # frobenius norm sqrt(sum(w**2))
    w = w / wnorm
    deltafg_tilde = jnp.einsum("f,tf->t", w, deltafg)
    signal_tilde = jnp.einsum("f,tf->t", w, signal)
    signal_tilde = signal_tilde.mean()
    # return (jnp.var(deltafg_tilde) / signal_tilde**2) + 1e10 * (wnorm**2 - 1) ** 2
    return (jnp.var(deltafg_tilde) / signal_tilde**2) + (wnorm**2 - 1) ** 2


def jax_optimize(lossfn, wtensor, deltafg, signal, learning_rate, niter):
    print("loading optimizer..")
    optimizer = optax.adam(learning_rate)
    schedule = optax.exponential_decay(
        init_value=learning_rate, transition_steps=niter / 10, decay_rate=0.99
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

    print("optimizing..")
    print("  ", f"{niter=}", f"{learning_rate=}")
    for i in tqdm(range(niter)):
        witers[i] = wtensor.copy()
        wnorms[i] = jnp.linalg.norm(wtensor)
        wtensor, opt_state, loss_value = _step(wtensor, opt_state)
        iterations[i], loss_vals[i] = i, loss_value
    woptim = wtensor
    return woptim, iterations, loss_vals, wnorms, witers


# %%
##--------------------------------------------------------------------##
# test multiple seeds


def run_optimizers(signal: str = "da", seed: int = 42, learning_rate=0.01, niter=50000):
    np.random.seed(seed)
    seed1, seed2, seed3 = np.random.randint(0, 1000, 3)
    print(f"  {seed1=}", f"{seed2=}", f"{seed3=}")
    jfg, jda, jcmb, jdelta, freqs = load_toysims(seed1, seed2)
    wtensor = get_wtensor(seed3)
    if signal == "da":
        print("  dark ages signal..")
        jsig = jda
    elif signal == "cmb":
        print("  cmb signal..")
        jsig = jcmb
    else:
        raise ValueError(f"signal {signal} not recognized, choose 'da' or 'cmb'")
    woptim, iterations, loss_vals, wnorms, witers = jax_optimize(
        lossfn, wtensor, jdelta, jsig, learning_rate, niter
    )
    return woptim, iterations, loss_vals, wnorms, witers


def run_seeds(nseeds, learning_rate, niter):
    out = dict()
    out["nseeds"] = nseeds
    out["learning_rate"] = learning_rate
    out["niter"] = niter
    for signal in ["da", "cmb"]:
        print(f"signal: {signal}")
        for i in range(nseeds):
            print(f"seed: {i}")
            woptim, iterations, loss_vals, wnorms, witers = run_optimizers(
                signal, i, learning_rate, niter
            )
            out[f"{signal}_{i}"] = (woptim, iterations, loss_vals, wnorms, witers)
    return out


def plot_seeds(freqs, wdict):
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    for isig, signal in enumerate(["da", "cmb"]):
        for i in range(wdict["nseeds"]):
            woptim, iterations, loss_vals, wnorms, witers = wdict[f"{signal}_{i}"]
            ax[isig, 0].plot(iterations, wnorms, label=f"seed {i}")
            ax[isig, 1].semilogy(iterations, loss_vals, label=f"seed {i}")
            ax[isig, 2].plot(freqs, woptim, label=f"seed {i}", color=f"C{i}")
        ax[isig, 0].set_title(f"{signal} wnorms")
        ax[isig, 0].set_xlabel("iterations")
        ax[isig, 0].legend()
        ax[isig, 1].set_title(f"{signal} loss")
        ax[isig, 1].set_xlabel("iterations")
        ax[isig, 1].legend()
        ax[isig, 2].set_title(f"{signal} woptim")
        ax[isig, 2].set_xlabel("frequency")
        ax[isig, 2].legend()
    plt.tight_layout()
    plt.show()


def animate_seeds(freqs, wdict):
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    frames = range(0, wdict["niter"], 100)

    def _update(frame):
        for isig, signal in enumerate(["da", "cmb"]):
            lnorms = ax[isig, 0].axvline(0)
            lloss = ax[isig, 1].axvline(0)
            (lweights,) = ax[isig, 2].plot([], [])
            for i in range(wdict["nseeds"]):
                woptim, iterations, loss_vals, wnorms, witers = wdict[f"{signal}_{i}"]
                # wnorms, loss, witers for da and cmb
                ax[isig, 0].plot(iterations, wnorms, label=f"seed {i}")
                ax[isig, 1].plot(iterations, loss_vals, label=f"seed {i}")
                ax[isig, 2].plot(
                    freqs, woptim, label=f"seed {i}", color=f"C{i}", linestyle="--"
                )
                lnorms.set_xdata([iterations[frame], iterations[frame]])
                lloss.set_xdata([iterations[frame], iterations[frame]])
                lweights.set_ydata(witers[frame])
                lims = max(abs(witers[frame].min()), abs(witers[frame].max()))
                ax[isig, 2].set_ylim(-lims, lims)
                ax[isig, 2].set_title(f"iteration {frame}")
        return lnorms, lloss, lweights

    ani = animation.FuncAnimation(fig, _update, frames=frames, interval=10)
    plt.show()


freqs = np.linspace(1, 50, 50)
wdict = run_seeds(10, 0.01, 50000)
plot_seeds(freqs, wdict)


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


# %%


# def likelihood(w, meanulsa, meansig, amp):
#     monopole = meanulsa + (1 - amp)[:, None] * meansig
#     return jnp.einsum("f,af->a", w, monopole)


# maxamp = 1e10
# amp = jnp.linspace(-maxamp, maxamp, 100)
# plt.plot(amp, likelihood(woptim, jfg.mean(axis=1), jda.mean(axis=1), amp))
# # plt.yscale("symlog")
# plt.show()


# # plot
# g = templates.plot.line(col="kind", sharey=False)
# g.axs[0, 0].set_yscale("log")
# plt.show()
# plt.figure(figsize=(12, 3))
# plt.subplot(131)
# fg.plot(norm=simutils.lognorm())
# plt.subplot(132)
# da.plot()
# plt.subplot(133)
# cmb.plot()
# plt.tight_layout()
# plt.show()

##--------------------------------------------------------------------##
# %%


# def loss1(w, deltafg):
#     wnorm = jnp.linalg.norm(w)  # frobenius norm sqrt(sum(w**2))
#     w = w / wnorm
#     deltafg_tilde = jnp.einsum("f,ft->t", w, deltafg)
#     return jnp.var(deltafg_tilde) + (wnorm - 1) ** 2


# %%

# %%
