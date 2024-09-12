# %%

import jax
import jax.numpy as jnp
import optax

# WARN: stupid jax doesn't like float64 by default
jax.config.update("jax_enable_x64", True)

import lusee
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xarray_einstats

import simflows.load as simloader
import simflows.pca as simpca
import simflows.utils as simutils

##--------------------------------------------------------------------##
# %%
"""
┌─────────────────────┐
│ 1. create mock sims │
└─────────────────────┘
"""

templates = simloader.load_templates(da_amp=1)
mock = simloader.load_mock_sim(da_amp=1)
fg, da, cmb, noise = mock
sum = mock.sum("kind")
# delta = sum - sum.mean("times")
delta = sum

# plot
g = templates.plot.line(col="kind", sharey=False)
g.axs[0, 0].set_yscale("log")
plt.show()
plt.figure(figsize=(12, 3))
plt.subplot(131)
fg.plot(norm=simutils.lognorm())
plt.subplot(132)
da.plot()
plt.subplot(133)
cmb.plot()
plt.tight_layout()
plt.show()

# delta.plot()
# plt.show()

##--------------------------------------------------------------------##
# %%


print("converting to jax arrays..")
delta, fg, da, cmb = jnp.array(delta), jnp.array(da), jnp.array(cmb), jnp.array(fg)

"""
┌─────────────────────────┐
│ 2. define loss function │
└─────────────────────────┘
"""


# def loss1(w, deltafg):
#     wnorm = jnp.linalg.norm(w)  # frobenius norm sqrt(sum(w**2))
#     w = w / wnorm
#     deltafg_tilde = jnp.einsum("f,ft->t", w, deltafg)
#     return jnp.var(deltafg_tilde) + (wnorm - 1) ** 2


def loss3(w, deltafg, signal=da):
    wnorm = jnp.linalg.norm(w)  # frobenius norm sqrt(sum(w**2))
    w = w / wnorm
    deltafg_tilde = jnp.einsum("f,ft->t", w, deltafg)
    signal_tilde = jnp.einsum("f,ft->t", w, signal)
    signal_tilde = signal_tilde.mean()
    # print(f"{signal_tilde=}")
    print("using var(deltafg)/signal**2 + (wnorm**2 - 1)**2 as loss..")
    return (jnp.var(deltafg_tilde) / signal_tilde**2) + (wnorm**2 - 1) ** 2
    # return jnp.var(deltafg_tilde) / signal_tilde**2


print("using loss3..")
lossfn = loss3

# %%

"""
┌──────────────────────────────────────────────┐
│ 3. init weights tensor and optimize wrt loss │
└──────────────────────────────────────────────┘
"""

seed = 0

print("generating random weights tensor..")
# the key to the life, universe and darkages
key = jax.random.PRNGKey(seed)
wtensor = jax.random.uniform(key, shape=(50))
wtensor = wtensor / jnp.linalg.norm(wtensor)
print(f"{wtensor.shape=}")
print(f"{jnp.linalg.norm(wtensor)=}")

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


# run the optimization loop
print("  optimizing..")
for i in range(1000):
    wtensor, opt_state, loss_value = step(wtensor, opt_state, delta)
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss_value}")
woptim = wtensor
print("done. have optimized parameters of shape:", woptim.shape)
print(f"{jnp.linalg.norm(woptim)=}")


plt.plot(mock.freqs, woptim)
plt.show()


##--------------------------------------------------------------------##
# %%
"""
┌─────────────────────────┐
│ 4. calculate likelihood │
└─────────────────────────┘
"""


def likelihood(w, meanulsa, meansig, amp):
    monopole = meanulsa + (1 - amp)[:, None] * meansig
    return jnp.einsum("f,af->a", w, monopole)


maxamp = 1e10
amp = jnp.linspace(-maxamp, maxamp, 100)
plt.plot(amp, likelihood(woptim, fg.mean(axis=1), da.mean(axis=1), amp))
# plt.yscale("symlog")
plt.show()


# %%
