# %%

from simflows.imports import *

# %%

ntimes = 650
idxs = simjax.random_normal((ntimes, 1), seed=0, mean=2.5, sigma=0.5)
amp20MHz = simjax.random_normal((ntimes, 1), seed=1, mean=1e4, sigma=1e4)
amp20MHz = jnp.abs(amp20MHz)
mock = simloader.load_mock_sim(idxs=idxs, amp20MHz=amp20MHz, da_amp=1)
fg, da, cmb, sum, delta = mock

# distribution of fg mock sim
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(idxs, amp20MHz)
ax[0].set_xlabel("spectral index")
ax[0].set_ylabel("amplitude at 20MHz")
fg.plot(
    x="freqs",
    hue="times",
    add_legend=False,
    yscale="log",
    color="grey",
    alpha=0.1,
    ax=ax[1],
)
ax[1].errorbar(fg.freqs, fg.mean("times"), fg.std("times"), fmt="o-", color="C0")
ax[1].set_yscale("symlog")
plt.show()

# distribution of delta
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
delta.plot(
    x="freqs",
    hue="times",
    add_legend=False,
    yscale="log",
    color="grey",
    alpha=0.1,
    ax=ax[0],
)
ax[0].plot(fg.freqs, fg.mean("times"), color="C0", label="fg.mean")
ax[0].errorbar(
    delta.freqs,
    delta.mean("times"),
    delta.std("times"),
    fmt="o-",
    color="C1",
    alpha=0.5,
    label="delta",
)
ax[0].set_yscale("symlog")
ax[0].legend()
ax[0].set_title("delta")
np.abs(delta).plot(
    x="freqs",
    hue="times",
    add_legend=False,
    yscale="log",
    color="grey",
    alpha=0.1,
    ax=ax[1],
)
ax[1].plot(fg.freqs, fg.mean("times"), color="C0", label="fg.mean")
ax[1].errorbar(
    delta.freqs,
    np.abs(delta).mean("times"),
    np.abs(delta).std("times"),
    fmt="o-",
    color="C1",
    alpha=0.5,
    label="abs(delta)",
)
ax[1].set_yscale("symlog")
ax[1].legend()
ax[1].set_title("abs(delta)")
plt.show()

# %%
##--------------------------------------------------------------------##
# as a time series

plt.figure(figsize=(13, 3))
plt.subplot(131)
plt.plot(idxs)
plt.title("spectral index")
plt.subplot(132)
plt.plot(amp20MHz)
plt.title("amplitude")
plt.subplot(133)
fg.plot(norm=simutils.lognorm(), x="times")
plt.tight_layout()
plt.show()

# %%
##--------------------------------------------------------------------##
# top pca components

delta_pca = simpca.get_pca(delta, "times", "freqs", other_dims=[])
proj = simpca.get_pca_proj(fg, da, cmb, delta_pca, "times", "freqs", [])

plt.figure(figsize=(10, 4))
plt.subplot(121)
delta_pca.U.sel(freqs_eig=[1.0, 2.0, 3.0]).plot.line(x="freqs", hue="freqs_eig")
plt.title("pca eigenvectors")
plt.subplot(122)
# delta_pca.S.plot(yscale="log", label="eigenvalues")
np.abs(proj.sel(kind="pfg").std("times")).plot(x="freqs_eig", label="rms pfg")
np.abs(proj.sel(kind="pfg").mean("times")).plot(x="freqs_eig", label="pfg")
np.abs(proj.sel(kind="pda").mean("times")).plot(x="freqs_eig", label="pda")
np.abs(proj.sel(kind="pcmb").mean("times")).plot(x="freqs_eig", label="pcmb")
# proj.sel(kind="pfg").plot( hue="times", x="freqs_eig", alpha=0.1, color="grey", add_legend=False)
# proj.sel(kind=["pfg", "pda", "pcmb"]).mean("times").plot( hue="kind", x="freqs_eig", yscale="symlog", add_legend=False)
# plt.title("pca projection")
plt.title("pca projection")
plt.yscale("symlog")
plt.legend()
plt.tight_layout()
plt.show()

# %%
##--------------------------------------------------------------------##
# %%
# what is the loss function value for each pca component?

loss3 = (
    proj.sel(kind="pfg").var("times").sum("freqs_eig")
    / proj.sel(kind="pda").mean("times").sum("freqs_eig") ** 2
)

# %%
##--------------------------------------------------------------------##
# what should the amp20 value be, s.t. fg.mean is within delta.std?

sigma_a20 = 1e3

idxs = simjax.random_normal((ntimes, 1), seed=0, mean=2.5, sigma=0.5)
amp20MHz = simjax.random_normal((ntimes, 1), seed=1, mean=1e4, sigma=sigma_a20)
amp20MHz = jnp.abs(amp20MHz)
mock = simloader.load_mock_sim(idxs=idxs, amp20MHz=amp20MHz, da_amp=1)
fg, da, cmb, sum, delta = mock
# distribution of mock fg sim
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(idxs, amp20MHz)
ax[0].set_xlabel("spectral index")
ax[0].set_ylabel("amplitude at 20MHz")
fg.plot(
    x="freqs",
    hue="times",
    add_legend=False,
    yscale="log",
    color="grey",
    alpha=0.1,
    ax=ax[1],
)
ax[1].errorbar(fg.freqs, fg.mean("times"), fg.std("times"), fmt="o-", color="C0")
ax[1].set_yscale("symlog")
plt.show()
# distribution of delta
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
delta.plot(
    x="freqs",
    hue="times",
    add_legend=False,
    yscale="log",
    color="grey",
    alpha=0.1,
    ax=ax[0],
)
ax[0].plot(fg.freqs, fg.mean("times"), color="C0", label="fg.mean")
ax[0].errorbar(
    delta.freqs,
    delta.mean("times"),
    delta.std("times"),
    fmt="o-",
    color="C1",
    alpha=0.5,
    label="delta",
)
ax[0].set_yscale("symlog")
ax[0].legend()
ax[0].set_title("delta")
np.abs(delta).plot(
    x="freqs",
    hue="times",
    add_legend=False,
    yscale="log",
    color="grey",
    alpha=0.1,
    ax=ax[1],
)
ax[1].plot(fg.freqs, fg.mean("times"), color="C0", label="fg.mean")
ax[1].errorbar(
    delta.freqs,
    np.abs(delta).mean("times"),
    np.abs(delta).std("times"),
    fmt="o-",
    color="C1",
    alpha=0.5,
    label="abs(delta)",
)
ax[1].set_yscale("symlog")
ax[1].legend()
ax[1].set_title("abs(delta)")
plt.show()


# %%


# %%

# %%

mock.sel(kind=["fg", "sum", "delta"]).plot(
    col="kind", x="times", norm=simutils.symlognorm()
)
plt.show()

g = mock.sel(kind=["fg", "sum"]).mean("times").plot(col="kind", x="freqs", yscale="log")
plt.show()

g = (
    mock.sel(kind=["fg", "delta"])
    .mean("times")
    .plot(col="kind", x="freqs", yscale="symlog")
)
plt.show()

plt.subplot(121)
fg.plot(norm=simutils.lognorm(), x="times")
plt.subplot(122)
mock["delta"].plot(x="times", norm=simutils.symlognorm())
plt.tight_layout()
plt.show()


# %%
# %%
# templates = loader.load_templates(freqs=jnp.linspace(1, 50, 393), da_amp=4e6)
templates = simloader.load_templates(da_amp=1)
mock = simloader.load_mock_sim(da_amp=1)
fg, da, cmb, noise = mock

g = templates.plot.line(row="kind", sharey=False)
g.axs[0, 0].set_yscale("log")
plt.show()

plt.figure(figsize=(13, 3))
plt.subplot(131)
fg.plot(norm=simutils.lognorm())
plt.subplot(132)
da.plot()
plt.subplot(133)
cmb.plot()
plt.show()
noise.plot()
plt.show()


# %%

# %%
##--------------------------------------------------------------------##
# %%
"""
┌─────────────────────────────────┐
│ individual pca of the mock sims │
└─────────────────────────────────┘
"""

delta_pca = simpca.get_pca(mock, "times", "freqs", other_dims=[])

delta_pca.S.plot(col="kind", yscale="log")
plt.show()

delta_pca.U.sel(
    freqs_eig=[
        1.0,
        2.0,
        3.0,
    ]
).plot.line(col="kind", x="freqs")
plt.show()

# mpca.Vt.plot(col="kind")
# plt.show()

# %%
##--------------------------------------------------------------------##
# %%

sum = mock.sum("kind")
sum.mean("times").plot(yscale="log")
plt.show()

spca = simpca.get_pca(sum, "times", "freqs", [])

spca.S.plot(yscale="log")
plt.show()

spca.U.sel(
    freqs_eig=[
        1.0,
        2.0,
        3.0,
    ]
).plot.line(x="freqs")
plt.show()

# spca.Vt.plot()
# plt.show()
# %%

spca.U.plot()
plt.show()

spca.Vt.plot()
plt.show()

# spca.to_dataarray(dim='pca').plot(col='pca')
# plt.show()


# ## analyze mock fg shapes using PCA
#
# ## status
# - [ ] TODO: task 1: description of the first task.
# - [ ] FIX: task 2: description of the second task.
# - [x] task 3: description of the third task.
#
# ## notes
# - important note 1.
# - WARN: important note 2.
# - BUG: important note 3.

# %%

from simflows.imports import *

ntimes = 650
idxs = simjax.random_normal((ntimes, 1), seed=0, mean=2.5, sigma=0.5)
amp20MHz = simjax.random_normal((ntimes, 1), seed=1, mean=1e4, sigma=1e4)
amp20MHz = jnp.abs(amp20MHz)
mock = simloader.load_mock_sim(idxs=idxs, amp20MHz=amp20MHz, da_amp=1)
fg, da, cmb, sum, delta = mock

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(idxs, amp20MHz)
ax[0].set_xlabel("spectral index")
ax[0].set_ylabel("amplitude at 20MHz")
fg.plot(
    x="freqs",
    hue="times",
    add_legend=False,
    yscale="log",
    color="grey",
    alpha=0.1,
    ax=ax[1],
)
ax[1].errorbar(fg.freqs, fg.mean("times"), fg.std("times"), fmt="o-", color="C0")
ax[1].set_yscale("symlog")
plt.show()


# %%

# g=mock.sel(kind=['fg','delta']).mean('times').plot(col='kind',x="freqs", yscale='log')
# plt.show()

# %%

plt.figure(figsize=(13, 3))
plt.subplot(131)
plt.plot(idxs)
plt.title("spectral index")
plt.subplot(132)
plt.semilogy(amp20MHz)
plt.title("amplitude")
plt.subplot(133)
fg.plot(norm=simutils.lognorm(), x="times")
plt.tight_layout()
plt.show()


# %%

mock.sel(kind=["fg", "sum", "delta"]).plot(
    col="kind", x="times", norm=simutils.symlognorm()
)
plt.show()

g = mock.sel(kind=["fg", "sum"]).mean("times").plot(col="kind", x="freqs", yscale="log")
plt.show()

g = (
    mock.sel(kind=["fg", "delta"])
    .mean("times")
    .plot(col="kind", x="freqs", yscale="symlog")
)
plt.show()

plt.subplot(121)
fg.plot(norm=simutils.lognorm(), x="times")
plt.subplot(122)
mock["delta"].plot(x="times", norm=simutils.symlognorm())
plt.tight_layout()
plt.show()


# %%
