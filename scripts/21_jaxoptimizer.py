##--------------------------------------------------------------------##
# %%

import jax

# WARN: this dumb shit is jax because they dont like float64. it only works at startup
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import xarray as xr

import simflows.utils as simutils

##--------------------------------------------------------------------##
# %%

# load sim, shape = (times,combs,freqs)
sim = xr.open_dataset("data/netcdf/sim.nc").sel(angles=0, combs=simutils.combmat)
ulsa = jnp.array(sim["ulsa"].data)
da = jnp.array(sim["da"].data)
cmb = jnp.array(sim["cmb"].data)

# means, shape = (combs,freqs)
meanulsa = ulsa.mean(axis=0)
meanda = da.mean(axis=0)
meancmb = cmb.mean(axis=0)

# delta fg
delta = ulsa - meanulsa
print(f"{ulsa.shape=}", f"{delta.shape=}")
print(f"{ulsa.var()=}", f"{delta.var()=}", f"{da.var()=}", f"{cmb.var()=}")

# %%

##--------------------------------------------------------------------##
# now we define a matrix of shape (combs,freqs) to optimize
# %%

# wtensor = jnp.ones((16, 50))
# the key to the life, universe and darkages
key = jax.random.PRNGKey(42)
wtensor = jax.random.uniform(key, shape=(16, 50))
wtensor = wtensor / jnp.linalg.norm(wtensor)
print(f"{wtensor.shape=}")
print(f"{jnp.linalg.norm(wtensor)=}")

wdelta = jnp.einsum("cf,tcf->t", wtensor, delta)
wda = jnp.einsum("cf,tcf->t", wtensor, da)
wcmb = jnp.einsum("cf,tcf->t", wtensor, cmb)
print(f"{wdelta.shape=}", f"{wda.shape=}", f"{wcmb.shape=}")
print(f"{wdelta.var()=}", f"{wda.var()=}", f"{wcmb.var()=}")

# ok, so this transforms them to a time series
# %%

##--------------------------------------------------------------------##
# now we define the loss function
# einsum is a godsend because it works with jax jit
# %%

# NOTE:
# now we have     ulsa,delta,da,cmb            (times,combs,freqs)
# means           ulsa.mean, da.mean, cmb.mean (combs,freqs)
# W reduced       wdelta,wda,wcmb              (times)
# interested in reducing the variance of wdelta and maximizing the variance of wda and/or wcmb
# but arent they constant in time?

plt.plot(wdelta, label="delta ulsa")
plt.plot(wda, label="da")
plt.plot(wcmb, label="cmb")
plt.yscale("symlog")
plt.legend()
plt.xlabel("time")
plt.show()

# %%
##--------------------------------------------------------------------##
# %%

# now we need a loss function


def loss1(w, deltafg):
    # we would like to minimize the variance of the reduced deltafg
    assert w.shape == (16, 50)
    assert deltafg.shape == (650, 16, 50)
    deltafg_tilde = jnp.einsum("cf,tcf->t", w, deltafg)
    return jnp.var(deltafg_tilde)


def loss2(w, deltafg, signal=meanda):
    # we would like to minimize the variance of the reduced deltafg
    assert w.shape == (16, 50)
    assert deltafg.shape == (650, 16, 50)
    assert signal.shape == (16, 50)
    deltafg_tilde = jnp.einsum("cf,tcf->t", w, deltafg)
    signal_tilde = jnp.einsum("cf,cf->", w, meanda)
    print(f"{signal_tilde=}")
    return jnp.var(deltafg_tilde) / signal_tilde**2


def loss3(w, deltafg, signal=meanda):
    # we would like to minimize the variance of the reduced deltafg
    assert w.shape == (16, 50)
    assert deltafg.shape == (650, 16, 50)
    assert signal.shape == (16, 50)
    wnorm = jnp.linalg.norm(w)  # frobenius norm sqrt(sum(w**2))
    w = w / wnorm
    deltafg_tilde = jnp.einsum("cf,tcf->t", w, deltafg)
    signal_tilde = jnp.einsum("cf,cf->", w, signal)
    # print(f"{signal_tilde=}")
    return (jnp.var(deltafg_tilde) / signal_tilde**2) + (wnorm**2 - 1) ** 2
    # return jnp.var(deltafg_tilde) / signal_tilde**2


# %%

##--------------------------------------------------------------------##
# %%
# now we use optax to define the optimizer

optimizer = optax.adam(1e-2)
opt_state = optimizer.init(wtensor)


# Define the update step
@jax.jit
def step(w, opt_state, deltafg):
    loss_value, grads = jax.value_and_grad(loss3)(w, deltafg)
    updates, opt_state = optimizer.update(grads, opt_state)
    w = optax.apply_updates(w, updates)
    return w, opt_state, loss_value


# Run the optimization loop
for i in range(1000):
    wtensor, opt_state, loss_value = step(wtensor, opt_state, delta)
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss_value}")
woptim = wtensor
print("done. have optimized parameters of shape:", woptim.shape)
print(f"{jnp.linalg.norm(woptim)=}")

# %%

##--------------------------------------------------------------------##
# %%


def likelihood(wfinal, meanulsa, meansig, amp):
    acf = meanulsa + (1 - amp)[:, None, None] * meansig
    return jnp.einsum("cf,acf->a", wfinal, acf)


maxamp = 1e10
amp = jnp.linspace(-maxamp, maxamp, 100)
plt.plot(amp, likelihood(woptim, meanulsa, meanda, amp))
# plt.yscale("symlog")
plt.show()

# %%

# %%
