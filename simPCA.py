import simutils
import numpy as np
import scipy
import lusee
import pickle

##---------------------------------------------------------------------------##


class SimData:
    """Wrapper for lusee.Data. Low-level interface to simulation fits file"""

    def __init__(self, path):
        self.sim = lusee.Data(path)

    def __getitem__(self, request):
        """request is a slice of the form [time,comb,freq]
        just like lusee.Data"""
        return self.sim[request]

    def set_comb(self, comb):
        """comb can be 'all', 'auto', or comb of the form '00R' etc."""
        ntimes, ncombs, nfreqs = self.sim.data.shape
        if comb == "all":
            raise NotImplementedError
        if comb == "auto":
            self.data = np.hstack(
                [
                    self.sim[:, "00R", :],
                    self.sim[:, "11R", :],
                    self.sim[:, "22R", :],
                    self.sim[:, "33R", :],
                ]
            ).T
            return self.data  # (4*nfreqs, ntimes)
        if simutils.is_comb(comb):
            self.data = self.sim[:, comb, :].T
            return self.data  # (nfreqs, ntimes)

    def subsample(self, n):
        self.data = self.data[:, ::n]


class SimForeground(SimData):
    """Wrapper for foreground data. Single fits file"""

    def __init__(self, path):
        print("loading foreground", path)
        super().__init__(path)

    def doPCA(self):
        print("doing foreground PCA")
        self.nfreqs, self.ntimes = self.data.shape  # (nfreqs, ntimes)
        self.mean = self.data.mean(axis=1)
        self.delta_data = self.data - self.mean[:, None]
        self.eve, self.eva, _ = scipy.linalg.svd(self.delta_data, full_matrices=False)
        assert self.eve.shape == (self.nfreqs, self.nfreqs)
        self.proj_data = self.eve.T @ self.delta_data
        self.proj_mean = self.eve.T @ self.mean
        self.proj_rms = np.sqrt(self.proj_data.var(axis=1))
        self.norm_pdata = self.delta_data / self.proj_rms[:, None]  # (nfreqs, ntimes)
        self.norm_pmean = self.proj_mean / self.proj_rms


class SimSignal(SimData):
    """Wrapper for signal data. Single fits file"""

    def __init__(self, path):
        print("loading signal", path)
        super().__init__(path)

    def project(self, fg):
        print("projecting signal")
        self.fg = fg
        self.mean = self.data.mean(axis=1)
        self.proj_mean = self.fg.eve.T @ self.mean
        self.norm_pmean = self.proj_mean / self.fg.proj_rms


class Foregrounds(SimForeground):
    """Wrapper for combined foregrounds. Multiple fits files"""

    def __init__(self, fgpaths):
        print("initializing combined foregrounds", fgpaths)
        self.fgs = [SimForeground(path) for path in fgpaths]

    def set_comb(self, comb):
        for fg in self.fgs:
            fg.data = fg.set_comb(comb)  # (ncombs*nfreqs, ntimes)
        self.data = np.hstack(
            [fg.data for fg in self.fgs]
        )  # (ncombs*nfreqs, nfgs*ntimes)
        self.nfreqs, self.ntimes = self.data.shape


class Signals(SimSignal):
    """Wrapper for combined signals. Multiple fits files"""

    def __init__(self, paths):
        print("initializing combined signals", paths)
        self.sigs = [SimSignal(path) for path in paths]

    def set_comb(self, comb):
        for sig in self.sigs:
            sig.data = sig.set_comb(comb)  # (ncombs*nfreqs, ntimes)
        self.data = np.hstack(
            [sig.data for sig in self.sigs]
        )  # (ncombs*nfreqs, nsigs*ntimes)
        self.nfreqs, self.ntimes = self.data.shape


class SkyAnalyzer:
    def __init__(self, config_paths):
        print("initializing sky analyzer foregrounds and signals..")
        self.ulsapaths, self.dapaths, self.cmbpaths = self.get_paths(config_paths)
        self.ulsa = Foregrounds(self.ulsapaths)
        self.cmb = Signals(self.cmbpaths)
        self.da = Signals(self.dapaths)

    def get_paths(self, config_paths):
        self.config_paths = config_paths
        self.configs = [simutils.Config(path) for path in self.config_paths]
        self.ulsapaths = [c.outdir + "/ulsa.fits" for c in self.configs]
        self.dapaths = [c.outdir + "/da.fits" for c in self.configs]
        self.cmbpaths = [c.outdir + "/cmb.fits" for c in self.configs]
        return self.ulsapaths, self.dapaths, self.cmbpaths

    def set_comb(self, comb):
        print("setting comb", comb)
        self.ulsa.set_comb(comb)
        self.da.set_comb(comb)
        self.cmb.set_comb(comb)
        return self.ulsa.data, self.da.data, self.cmb.data

    def subsample(self, n):
        print("subsampling to", n, "min")
        self.ulsa.subsample(n)
        self.da.subsample(n)
        self.cmb.subsample(n)
        return self.ulsa.data, self.da.data, self.cmb.data

    def doPCA_and_project(self):
        print("doing PCA and projecting")
        self.ulsa.doPCA()
        self.da.project(self.ulsa)
        self.cmb.project(self.ulsa)
        return self.ulsa, self.da, self.cmb

    def save(self, outname):
        print("saving to", outname)
        with open(outname, "wb") as f:
            pickle.dump(self, f)


##---------------------------------------------------------------------------##
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys
    import seaborn as sns
    import pandas as pd

    if len(sys.argv) < 2:
        print("Usage: python simPCA.py <config_paths>")
        sys.exit(1)
    config_paths = sys.argv[1:]

    print("loading sky..")
    sky = SkyAnalyzer(config_paths)
    # sky.save("outputs/sky.pkl")
    # sky = pickle.load(open("outputs/sky.pkl", "rb"))

    print("plot 00R..")
    plt.figure(figsize=(8, 8))
    sky.set_comb("00R")
    sky.subsample(60)
    sky.doPCA_and_project()
    simutils.plt_scree(sky, ax=plt.gca(), alpha=alphas[si])
    plt.plot([], [], "C0", label="mean ulsa")
    plt.plot([], [], "C1", label="mean da")
    plt.plot([], [], "C2", label="mean cmb")
    plt.plot([], [], "C3", label="rms ulsa")
    plt.title(f"00R, {sim_paths}", fontsize="x-small")
    plt.legend()
    plt.show()

    # print("plot auto..")
    # plt.figure(figsize=(8, 8))
    # alphas = [1, 0.7, 0.5, 0.3]
    # for si, subsample in enumerate([1, 10, 30, 60]):
    #     print(f"subsample: {subsample}")
    #     sky.set_comb("auto")
    #     sky.subsample(subsample)
    #     sky.doPCA_and_project()
    #     simutils.plt_scree(sky, ax=plt.gca(), alpha=alphas[si])
    #     plt.plot([], [], "gray", label=f"{subsample} min", alpha=alphas[si])
    # plt.plot([], [], "C0", label="mean ulsa")
    # plt.plot([], [], "C1", label="mean da")
    # plt.plot([], [], "C2", label="mean cmb")
    # plt.plot([], [], "C3", label="rms ulsa")
    # plt.title(f"auto, {sim_paths}", fontsize="x-small")
    # plt.legend()
    # plt.show()

    print("plot pair plot..")
    sky.set_comb("00R")
    sky.subsample(60)
    sky.doPCA_and_project()
    eigmodes = [0, 10, 20, 30, 40]
    _, ndata = sky.ulsa.norm_pdata.shape
    d = np.vstack(
        [
            sky.ulsa.norm_pdata.T,
            sky.ulsa.norm_pmean,
            sky.da.norm_pmean,
            sky.cmb.norm_pmean,
        ]
    )
    index = ["data"] * ndata + ["mean ulsa", "mean da", "mean cmb"]
    df = pd.DataFrame(d, index=index).reset_index()
    kwargs = {
        "markers": [".", "d", "^", "v"],
        "height": 1,
        "vars": eigmodes,
        "hue": "index",
    }
    sns.pairplot(df, **kwargs)
    plt.show()

###---------------------------------------------------------------------------##
