# %%

import jax

# WARN: this dumb shit is jax because they dont like float64. it only works at startup
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

arr = jnp.arange(10).astype(jnp.float64)
print(arr.dtype)


from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform)

# %%


# %%
