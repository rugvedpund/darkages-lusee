import lusee
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse

def timeit(func):
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        print(f"{func.__name__} took {(time.time() - start):.1f} seconds")
        return out

    return wrapper

def exp(arr: np.ndarray):
    return np.exp(arr - arr.max())

def is_comb(comb):
    assert comb[-1] in ["R", "I"]
    assert len(comb) == 3
    return True


def get_configname(config):
    pass


def flatten_combs(data):
    ntimes, ncombs, nfreqs = data.shape  # (ntimes, ncombs, nfreqs)
    data = np.transpose(data, (1, 0, 2))  # (ncombs, ntimes, nfreqs)
    data = data.reshape(ncombs * ntimes, nfreqs)  # (ncombs*ntimes, nfreqs)
    data = data.T  # (nfreqs, ncombs*ntimes)
    return data


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", help="config yaml paths")
    parser.add_argument("--outputs", nargs="*", help="output dir path")
    parser.add_argument("--params_yaml", type=str, help="param yaml path")
    parser.add_argument("--retrain", action="store_true")
    return parser


def combs(nbeams=4):
    if nbeams != 4:
        raise NotImplementedError
    return [
        "00R",
        "01R",
        "01I",
        "02R",
        "02I",
        "03R",
        "03I",
        "11R",
        "12R",
        "12I",
        "13R",
        "13I",
        "22R",
        "23R",
        "23I",
        "33R",
    ]


def all_combs(n):
    combs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                combs.append(f"{i}{j}R")
            if i > j:
                combs.append(f"{j}{i}R")
            if i < j:
                combs.append(f"{i}{j}I")
    return combs


def plt_waterfall(D, comb, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(
        D[:, comb, :].T,
        aspect="auto",
        # extent=(D.freq[0], D.freq[-1], len(D.times), 0),
        origin="lower",
        cmap="viridis",
        **kwargs,
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.ax.tick_params(labelsize=8)
    ax.set_ylabel("f(MHz)")
    ax.set_xlabel("time")
    ax.set_title(f"{comb}")
    return ax


def plt_scree(sky, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.plot(np.abs(sky.ulsa.proj_mean), c="C0", **kwargs)
    ax.plot(np.abs(sky.da.proj_mean), c="C1", **kwargs)
    ax.plot(np.abs(sky.cmb.proj_mean), c="C2", **kwargs)
    ax.plot(sky.ulsa.proj_rms, c="C3", **kwargs)
    ax.set_xlabel("eigmodes")
    ax.set_ylabel("T[K]")
    ax.set_yscale("log")
    ax.grid()
    return ax


# D = lusee.Data("gaussbeam.fits")
# fig, ax = plt.subplots(4, 4, figsize=(15, 8))
# for comb, ax in zip(all_combs(4), ax.flatten()):
#     plt_waterfall(D, comb, ax=ax, norm=mcolors.SymLogNorm(linthresh=1e2))
# fig.suptitle(r"NSEW Gaussian $10^\circ$ Beams at $40^\circ$ declination")
# fig.tight_layout()
# plt.show()

# D = lusee.Data("realistic_example.fits")
# fig, ax = plt.subplots(4, 4, figsize=(15, 8))
# for comb, ax in zip(all_combs(4), ax.flatten()):
#     plt_waterfall(D, comb, ax=ax, norm=mcolors.SymLogNorm(linthresh=1e2))
# fig.suptitle("NSEW LuSEE Beams")
# fig.tight_layout()
# plt.show()

# D = lusee.Data("smallgaussbeam.fits")
# fig, ax = plt.subplots(4, 4, figsize=(15, 8))
# for comb, ax in zip(all_combs(4), ax.flatten()):
#     plt_waterfall(D, comb, ax=ax, norm=mcolors.SymLogNorm(linthresh=1e2))
# fig.suptitle(r"NSEW Gaussian $2^\circ$ Beams at $40^\circ$ declination")
# fig.tight_layout()
# plt.show()
