##--------------------------------------------------------------------##
# %%

import funfact as ff
import jax.numpy as jnp
import xarray as xr

import simflows.utils as simutils

# %%


sim = xr.open_dataset("data/netcdf/sim.nc").sel(angles=0, combs=simutils.combmat)
#  NOTE: shape = (times,combs,freqs)
ulsa = jnp.array(sim["ulsa"].data)
delta = ulsa - ulsa.mean(axis=0)  # mean subtracted
da = jnp.array(sim["da"].data)
cmb = jnp.array(sim["cmb"].data)
print(f"{ulsa.shape=}", f"{delta.shape=}")
print(f"{ulsa.var()=}", f"{delta.var()=}", f"{da.var()=}", f"{cmb.var()=}")
# %%

##--------------------------------------------------------------------##
# %%

target = ulsa + da + cmb
print(f"{target.shape=}")

# %%
##--------------------------------------------------------------------##
# create the abstract tensors for decomposition
# %%

tbeams = ff.tensor("tbeams", 16, 50, 100)
tcmb = ff.tensor("tcmb", 1, prefer=ff.conditions.NonNegative)
dummy = ff.tensor("dummy", jnp.ones((650, 50, 100)))
tda = ff.tensor("tda", 50)
tulsa = ff.tensor("tulsa", 650, 50, 100)
c, f, l, t = ff.indices("c f l t")  # combs, freqs, ell sph-harm, times

tsrex = tulsa[t, ~f, l] * tbeams[c, ~f, l]
tsrex = tsrex >> [t, c, f]

# tsrex = (
#     dummy[~t, ~f, l] * tbeams[~c, f, l] * tcmb
#     + dummy[~t, ~f, l] * tbeams[c, ~f, l] * tda[f]
#     + tulsa[~t, ~f, l] * tbeams[~c, ~f, l]
# )

# %%
##--------------------------------------------------------------------##
# %%

fac = ff.Factorization.from_tsrex(tsrex, target)

# %%
