# %%

from simflows.imports import *

##--------------------------------------------------------------------##
# %%

ntimes = 650
idxs = simjax.random_normal((ntimes, 1), seed=0, mean=2.5, sigma=0.5)
amp20MHz = simjax.random_normal((ntimes, 1), seed=1, mean=1e4, sigma=1e4)
amp20MHz = jnp.abs(amp20MHz)
mock = simloader.load_mock_sim(idxs=idxs, amp20MHz=amp20MHz, da_amp=1)
fg, da, cmb, sum, delta = mock

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
jda = jda.at[-1].set(1e-4)

# %%
##--------------------------------------------------------------------##
# loss func


def loss3(w, deltafg, signal=jda):
    wnorm = jnp.linalg.norm(w)  # frobenius norm sqrt(sum(w**2))
    w = w / wnorm
    deltafg_tilde = jnp.einsum("f,tf->t", w, deltafg)
    signal_tilde = jnp.einsum("f,tf->t", w, signal)
    signal_tilde = signal_tilde.mean()
    print("using var(deltafg)/signal**2 + (wnorm**2 - 1)**2 as loss..")
    # return jnp.var(deltafg_tilde) / signal_tilde**2
    return (jnp.var(deltafg_tilde) / signal_tilde**2) + 1e10 * (wnorm**2 - 1) ** 2
    # return (jnp.var(deltafg_tilde) / signal_tilde**2) + (wnorm**2 - 1) ** 4
    # return (jnp.var(deltafg_tilde) / signal_tilde**2) + (wnorm**2 - 1) ** 10


# print(loss3(np.ones(50), jdelta, jda))


print("using loss3..")
lossfn = loss3

seed = np.random.randint(0, 1e6)
# the key to the life, universe and darkages
print("generating random weights tensor..")
key = jax.random.PRNGKey(seed)
wtensor = jax.random.uniform(key, shape=(50)) - 0.5
# wtensor = jnp.zeros(50)
# wtensor = wtensor.at[1].set(1)
print(f"{jnp.linalg.norm(wtensor)=}")
wtensor = wtensor / jnp.linalg.norm(wtensor)
print(f"{wtensor.shape=}")
print(f"{jnp.linalg.norm(wtensor)=}")


plt.plot(mock.freqs, wtensor)
plt.title(f"init seed {seed}, norm={jnp.linalg.norm(wtensor):.2f}")
plt.show()


# jax recommends optax for optimizers
print("init optimizer..")
optimizer = optax.adam(1e-2)
opt_state = optimizer.init(wtensor)


# define the update step
@jax.jit
def step(w, opt_state, deltafg):
    loss_value, grads = jax.value_and_grad(lossfn)(w, deltafg)
    updates, opt_state = optimizer.update(grads, opt_state)
    w = optax.apply_updates(w, updates)
    return w, opt_state, loss_value


print("ready..")


# run the optimization loop
print("  optimizing..")
niter = 100000
iterations = np.zeros((niter))
loss_vals = np.zeros((niter))
wnorms = np.zeros((niter))
witers = np.zeros((niter, 50))

for i in range(niter):
    witers[i] = wtensor.copy()
    wnorms[i] = jnp.linalg.norm(wtensor)
    wtensor, opt_state, loss_value = step(wtensor, opt_state, jdelta)
    iterations[i], loss_vals[i] = i, loss_value
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss_value}")
woptim = wtensor
print("done..")
print("have woptim of shape:", woptim.shape)
print(f"  and {jnp.linalg.norm(woptim)=}")

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


frames = range(0, niter, 10)

# for i in frames:
#     plt.plot(mock.freqs, witers[i], color=mpl.cm.viridis(i / niter))
# plt.plot(mock.freqs, woptim)
# plt.show()


fig, ax = plt.subplots()
(line,) = ax.plot(mock.freqs, witers[0], color="blue")

# Set plot limits
ax.set_xlim(mock.freqs.min(), mock.freqs.max())
ax.set_xlabel("Frequency")
ax.set_ylabel("Weight")


# Update function to update the plot for each frame
def update(frame):
    line.set_ydata(witers[frame])
    lims = max(abs(witers[frame].min()), abs(witers[frame].max()))
    ax.set_ylim(-lims, lims)
    ax.set_title(f"Iteration {frame}")
    return (line,)


# Create the animation
ani = mpl.animation.FuncAnimation(fig, update, frames=frames, interval=10)
plt.show()
# ani.save('optimization_progress.mp4', writer='ffmpeg')


# %%


def likelihood(w, meanulsa, meansig, amp):
    monopole = meanulsa + (1 - amp)[:, None] * meansig
    return jnp.einsum("f,af->a", w, monopole)


maxamp = 1e10
amp = jnp.linspace(-maxamp, maxamp, 100)
plt.plot(amp, likelihood(woptim, jfg.mean(axis=1), jda.mean(axis=1), amp))
# plt.yscale("symlog")
plt.show()


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
