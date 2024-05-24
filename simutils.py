import lusee
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import yaml
import os


class Config(dict):
    def __init__(self, yaml_file="configs/simtemplate.yaml"):
        print("Loading config from ", yaml_file)
        self.yaml_file = yaml_file
        self.from_yaml(self.yaml_file)

    def __repr__(self):
        return yaml.dump(self)

    def __str__(self):
        return yaml.dump(self)

    @property
    def name(self):
        yamlnamedotyaml = os.path.basename(self.yaml_file)
        yamlname = os.path.splitext(yamlnamedotyaml)[0]
        return yamlname

    @property
    def outdir(self):
        return os.path.join(os.environ["LUSEE_OUTPUT_DIR"], self.name)

    def from_yaml(self, yaml_file):
        with open(yaml_file) as f:
            yml = yaml.safe_load(f)
        self.update(yml)

    def make_ulsa_config(self):
        print("Making ULSA config")
        self["sky"] = {"file": "ULSA_32_ddi_smooth.fits"}
        self["simulation"][
            "output"
        ] = f"{self.name}/ulsa.fits"  # $LUSEE_OUTPUT_DIR + self.name + "/ulsa.fits"
        return self

    def make_da_config(self):
        print("Making DA config")
        self["sky"] = {"type": "DarkAges"}
        self["simulation"][
            "output"
        ] = f"{self.name}/da.fits"  # $LUSEE_OUTPUT_DIR + self.name + "/da.fits"
        return self

    def make_cmb_config(self):
        print("Making CMB config")
        self["sky"] = {"type": "CMB"}
        self["simulation"][
            "output"
        ] = f"{self.name}/cmb.fits"  # $LUSEE_OUTPUT_DIR + self.name + "/cmb.fits"
        return self


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
    parser.add_argument("fname", type=str)
    parser.add_argument("--comb", type=str)
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
