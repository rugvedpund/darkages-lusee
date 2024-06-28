import os
import sys
import pickle
import yaml
import numpy as np
import scipy
import lusee
import simutils
from sinf import GIS
import torch
torch.manual_seed(0)

##---------------------------------------------------------------------------##


class Params(dict):
    def __init__(self):
        pass

    def from_yaml(self, yaml_file):
        print("loading params from", yaml_file)
        with open(yaml_file) as f:
            yml = yaml.safe_load(f)
        self.update(yml)
        return self

    def __repr__(self):
        return yaml.dump(self)

    pass


class SimFlow:
    @simutils.timeit
    def __init__(self, args):
        self.args = args
        self.params = Params().from_yaml(args.params_yaml)
        self.load()

    def load(self):
        if os.path.exists(self.params["flow"]["path"]) and not self.args.retrain:
            print("loading simflow from", self.params["flow"]["path"])
            with open(self.params["flow"]["path"], "rb") as f:
                self.results = pickle.load(f)
                self.flow = self.results["flow"]
                self.sky = self.results["sky"]
                return self
        print("loading sky from scratch..")
        self.sky = self.load_sky()
        self.flow = self.load_flow()

    def load_sky(self):
        # pkl_path = self.params["sky"]["pkl_path"]
        # if os.path.exists(pkl_path):
        #     self.sky = SkyAnalyzer().from_pickle(pkl_path)
        #     return self.sky
        self.sky = SkyAnalyzer().from_configs(self.params["sky"]["config_paths"])
        self.sky.set_comb(self.params["sky"]["comb"])
        self.sky.subsample(self.params["sky"]["subsample"])
        self.sky.doPCA_and_project()
        return self.sky

    def load_flow(self):
        print("loading new normalizing flow..")
        return LikelihoodModel()

    @simutils.timeit
    def train(self):
        # need (ntimes, nfreqs) because GIS expects (ndata, ndim)
        print("training flow..")
        self.flow.train(self.sky.ulsa.norm_pdata.T.copy())  # (ntimes, nfreqs)

    def save(self):
        self.results = {}
        self.results["flow"] = self.flow
        self.results["sky"] = self.sky
        print("saving simflow to", self.params["flow"]["path"])
        with open(self.params["flow"]["path"], "wb") as f:
            pickle.dump(self.results, f)

    def save_flow(self):
        print("saving flow to", self.params["flow"]["path"])
        self.flow.save(self.params["flow"]["path"])

    def run(self, amp: np.ndarray, true_amp=1.0):
        print("running flow to calculate likelihoods..")
        self.amp = amp[:, None]  # (namps, 1)
        self.ulsa_means = self.sky.ulsa.norm_pmean.copy()[None, :]  # (1, nfreqs)
        self.true_da_means = (
            true_amp * self.sky.da.norm_pmean.copy()[None, :]
        )  # (1, nfreqs)
        self.true_cmb_means = (
            true_amp * self.sky.cmb.norm_pmean.copy()[None, :]
        )  # (1, nfreqs)

        self.ampxda_means = (
            self.amp * self.sky.da.norm_pmean.copy()[None, :]
        )  # (namps, nfreqs)
        self.ampxcmb_means = (
            self.amp * self.sky.cmb.norm_pmean.copy()[None, :]
        )  # (namps, nfreqs)
        self.da_loglikelihood = self.flow.likelihood(
            self.ulsa_means + self.true_da_means - self.ampxda_means
        )  # (namps, 1)
        self.cmb_loglikelihood = self.flow.likelihood(
            self.ulsa_means + self.true_cmb_means - self.ampxcmb_means
        )  # (namps, 1)
        return self.da_loglikelihood, self.cmb_loglikelihood


##---------------------------------------------------------------------------##


class LikelihoodModel:
    """Base class for any likelihood model"""

    def __init__(self):
        self.model = None

    def train(self, data):
        raise NotImplementedError

    def likelihood(self, vec):
        raise NotImplementedError


class SINF(LikelihoodModel):
    def __init__(self):
        super().__init__()
        return self

    def likelihood(self, vec):
        assert vec.shape[1] == self.ndim
        return self.toCPU(self.model.evaluate_density(self.toGPU(vec.copy())))

    def train(self, data):
        self.model = None
        self.ndata, self.ndim = data.shape
        frac = 0.8
        self.ntrain = int(frac * self.ndata)
        traindata = data[: self.ntrain, :].copy()
        valdata = data[self.ntrain :, :].copy()
        self.model = GIS.GIS(self.toGPU(traindata), self.toGPU(valdata), verbose=True)

    def load(self, model_path):
        print("loading model from", model_path)
        self.model = torch.load(model_path)
        self.ndim = self.model.ndim
        return self

    def toGPU(self, nparray):
        return torch.from_numpy(nparray).float().to(torch.device("cuda"))

    def save(self, model_path):
        print("saving model", model_path)
        torch.save(self.model, model_path)

    def toCPU(self, arr: torch.Tensor):
        return arr.cpu().detach().numpy()

    def forward(self, vec):
        assert vec.shape[1] == self.ndim
        fwd, _ = self.model.forward(self.toGPU(vec.copy()))
        return self.toCPU(fwd)


class Gaussian:
    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = cov

    def loglikelihood(self, vec):
        return scipy.stats.multivariate_normal.logpdf(vec, mean=self.mu, cov=self.cov)


class GaussianApprox(LikelihoodModel):
    def __init__(self):
        super().__init__()

    def train(self, sky):
        print("training Gaussian model..")
        self.sky = sky
        self.data = sky.ulsa.norm_pdata.copy()
        self.pmean = sky.ulsa.norm_pmean.copy()
        self.ndim, self.ndata = self.data.shape
        self.modes = np.linspace(1, self.ndim, num=self.ndim)
        self.mu = self.data.mean(axis=1)
        self.cov = np.cov(self.data)
        self.sigmas = np.sqrt(np.diag(self.cov))
        self.model = Gaussian(self.mu, self.cov)
        return self

    @property
    def sigmadev(self):
        return np.abs(self.pmean - self.mu) / self.sigmas

    def scale_model(self, scale):
        self.scale = scale
        self.cov = self.cov * self.scale[:, None] ** 2
        self.sigmas = np.sqrt(np.diag(self.cov))
        self.model = Gaussian(self.mu, self.cov)
        return self

    # def autoscale_model(self, kind="linear", **kwargs):
    #     print("autoscaling model..")
    #     if kind == "linear":
    #         return self.autoscale_linear(**kwargs)
    #     if kind == "sigmoid":
    #         return self.autoscale_sigmoid(**kwargs)
    #     else:
    #         raise NotImplementedError

    # def autoscale_linear(self, slope=1.0, intercept=0.0):
    #     print("using linear autoscale with slope", slope, "and intercept", intercept)
    #     nmodes = self.ndim
    #     scale = slope*self.modes + intercept
    #     self.cov = self.cov * scale[:,None]**2
    #     self.sigmas = np.sqrt(np.diag(self.cov))
    #     self.model = Gaussian(self.mu, self.cov)
    #     return self

    def get_da_loglikelihood(self, amp, true_amp=1.0):
        print("getting DA loglikelihood..")
        assert self.model is not None, "need to train model first"
        self.ulsa_means = self.sky.ulsa.norm_pmean.copy()[None, :]
        self.da_means = self.sky.da.norm_pmean.copy()[None, :] * true_amp
        self.ampxda_means = self.sky.da.norm_pmean.copy()[None, :] * amp[:, None]
        self.da_loglikelihood = self.model.loglikelihood(self.ulsa_means + self.da_means - self.ampxda_means)
        return self.da_loglikelihood

    def get_cmb_loglikelihood(self, amp, true_amp=1.0):
        print("getting CMB loglikelihood..")
        assert self.model is not None, "need to train model first"
        self.ulsa_means = self.sky.ulsa.norm_pmean.copy()[None, :]
        self.cmb_means = self.sky.cmb.norm_pmean.copy()[None, :] * true_amp
        self.ampxcmb_means = self.sky.cmb.norm_pmean.copy()[None, :] * amp[:, None]
        self.cmb_loglikelihood = self.model.loglikelihood(self.ulsa_means + self.cmb_means - self.ampxcmb_means)
        return self.cmb_loglikelihood


##---------------------------------------------------------------------------##

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
        return self

    def make_ulsa_config(self):
        print("Making ULSA config")
        self["sky"] = {"file": "ULSA_32_ddi_smooth.fits"}
        self["simulation"][
            "output"
        ] = f"{self.name}/ulsa.fits"  # $LUSEE_OUTPUT_DIR + self.name + "/ulsa.fits"
        return self

    def make_da_config(self):
        print("Making DA config")
        self["sky"] = {
            "type": "DarkAges",
            "scaled": True,
            "nu_min": 16.4,
            "nu_rms": 14.0,
            "A": 0.04,
        }
        self["simulation"][
            "output"
        ] = f"{self.name}/da.fits"  # $LUSEE_OUTPUT_DIR + self.name + "/da.fits"
        return self

    def make_cmb_config(self):
        print("Making CMB config")
        self["sky"] = {"type": "CMB", "Tcmb": 2.73}
        self["simulation"][
            "output"
        ] = f"{self.name}/cmb.fits"  # $LUSEE_OUTPUT_DIR + self.name + "/cmb.fits"
        return self

    def save(self, outname):
        print("saving config to", outname)
        with open(outname, "wb") as f:
            pickle.dump(self, f)


class SkyAnalyzer:
    def __init__(self):
        print("initializing sky analyzer foregrounds and signals..")

    def from_args(self, args):
        config_paths = args.configs
        output_paths = args.outputs
        if config_paths:
            return self.from_configs(config_paths)
        if output_paths:
            return self.from_outputs(output_paths)

    def from_pickle(self, pickle_path):
        print("loading sky from pickle", pickle_path)
        with open(pickle_path, "rb") as f:
            self = pickle.load(f)
        return self

    def from_outputs(self, output_dirs):
        print("loading sky from outputs", output_dirs)
        for output_dir in output_dirs:
            self.ulsapaths = [output_dir + "ulsa.fits"]
            self.dapaths = [output_dir + "da.fits"]
            self.cmbpaths = [output_dir + "cmb.fits"]
        self.ulsa = Foregrounds(self.ulsapaths)
        self.da = Signals(self.dapaths)
        self.cmb = Signals(self.cmbpaths)
        return self

    def from_configs(self, config_paths):
        print("loading sky from configs", config_paths)
        self.ulsapaths, self.dapaths, self.cmbpaths = self.get_paths(config_paths)
        self.ulsa = Foregrounds(self.ulsapaths)
        self.da = Signals(self.dapaths)
        self.cmb = Signals(self.cmbpaths)
        return self

    def get_paths(self, config_paths):
        self.config_paths = config_paths
        self.configs = [Config(path) for path in self.config_paths]
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
        print(
            "subsampling from (nfreq,ndata)",
            self.ulsa.data.shape,
            " to",
            self.ulsa.data[:, ::n].shape,
        )
        self.ulsa.subsample(n)
        self.da.subsample(n)
        self.cmb.subsample(n)
        return self.ulsa.data, self.da.data, self.cmb.data

    def add_noise(self, SNR):
        """noise is radiometer like"""
        self.noise = SimNoise(SNR)
        print(f"adding noise from SNR {SNR:.0e} to sky..")
        self.ulsa.data = self.noise.add_noise(self.ulsa.data)
        self.da.data = self.noise.add_noise(self.da.data)
        self.cmb.data = self.noise.add_noise(self.cmb.data)
        return self.ulsa.data, self.da.data, self.cmb.data

    def doPCA_and_project(self):
        print("doing PCA and projecting")
        self.ulsa.doPCA()
        self.da.project(self.ulsa)
        self.cmb.project(self.ulsa)
        return self.ulsa, self.da, self.cmb

    def save(self, outname):
        print("saving sky to", outname)
        with open(outname, "wb") as f:
            pickle.dump(self, f)


class SimData:
    """Low-level wrapper for lusee.Data. Low-level interface to simulation fits file"""

    def __init__(self, path):
        self.sim = lusee.Data(path)

    def __getitem__(self, request):
        """request is a slice of the form [time,comb,freq]
        just like lusee.Data"""
        return self.sim[request]

    def get_comb(self,comb):
        return self.sim[:,comb,:].T # (nfreqs, ntimes)

    def set_comb(self, comb):
        """comb can be 'all', 'auto', or comb of the form '00R' etc."""
        ntimes, ncombs, nfreqs = self.sim.data.shape
        if comb == "all":
            raise NotImplementedError
        if comb == "auto":
            raise NotImplementedError
            # self.data = np.hstack(
            #     [
            #         self.sim[:, "00R", :],
            #         self.sim[:, "11R", :],
            #         self.sim[:, "22R", :],
            #         self.sim[:, "33R", :],
            #     ]
            # ).T
            # return self.data  # (4*nfreqs, ntimes)
        if simutils.is_comb(comb):
            self.data = self.sim[:, comb, :].T
            return self.data  # (nfreqs, ntimes)
        raise NotImplementedError

    def subsample(self, n):
        self.data = self.data[:, ::n]
        return self


class SimForeground(SimData):
    """Wrapper for foreground data. Single fits file"""

    def __init__(self, path):
        print("loading foreground", path)
        super().__init__(path)

    def doPCA(self):
        print("doing foreground PCA")
        self.nfreqs, self.ntimes = self.data.shape  # (nfreqs, ntimes)
        # self.rng = np.random.default_rng(0)
        # self.data = self.rng.permutation(self.data, axis=1)
        self.mean = self.data.mean(axis=1)
        self.delta_data = self.data - self.mean[:, None]
        self.eve, self.eva, _ = scipy.linalg.svd(self.delta_data, full_matrices=False)
        # self.eva, self.eve = np.linalg.eig(self.delta_data @ self.delta_data.T)
        assert self.eve.shape == (self.nfreqs, self.nfreqs)
        self.proj_data = self.eve.T @ self.delta_data
        self.proj_mean = self.eve.T @ self.mean
        self.proj_rms = np.sqrt(self.proj_data.var(axis=1))
        self.norm_pdata = self.proj_data / self.proj_rms[:, None]  # (nfreqs, ntimes)
        self.norm_pmean = self.proj_mean / self.proj_rms
        return self


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
        return self


class SimNoise:
    def __init__(self, SNR):
        self.SNR = SNR
        self.noise = None

    def add_noise(self, data):
        """noise is radiometer like"""
        ndim, ntimes = data.shape
        noise_sigma = data.mean(axis=1) * np.sqrt(ntimes) / self.SNR
        self.noise = np.vstack(
            [np.random.normal(0, sigma, ntimes) for sigma in noise_sigma]
        )
        print("adding noise ", self.noise.shape, "to (nfreq,ndata)", data.shape)
        print(noise_sigma)
        print(self.noise)
        return data + self.noise


class Foregrounds(SimForeground):
    """Wrapper for combined foregrounds. Multiple fits files"""

    def __init__(self, fgpaths):
        print("initializing combined foregrounds..")
        self.fgs = [SimForeground(path) for path in fgpaths]
        # self.fgs = self.load_parallel(fgpaths)

    def get_comb(self,comb):
        for fg in self.fgs:
            fg.data = fg.get_comb(comb)

    def set_comb(self, comb):
        for fg in self.fgs:
            fg.data = fg.set_comb(comb)  # (ncombs*nfreqs, ntimes)
        self.data = np.hstack(
            [fg.data for fg in self.fgs]
        )  # (ncombs*nfreqs, nfgs*ntimes)
        self.nfreqs, self.ntimes = self.data.shape

    def load_parallel(self, fgpaths):
        print("loading fgs parallel from ", fgpaths)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            print("submitting tasks..")
            future_fgs = [executor.submit(SimForeground, path) for path in fgpaths]
            print("waiting for results..")
            self.fgs = [future.result() for future in future_fgs]
            print("done")
        return self.fgs


class Signals(SimSignal):
    """Wrapper for combined signals. Multiple fits files"""

    def __init__(self, paths):
        print("initializing combined signals..")
        self.sigs = [SimSignal(path) for path in paths]
        # self.sigs = self.load_parallel(paths)

    def set_comb(self, comb):
        for sig in self.sigs:
            sig.data = sig.set_comb(comb)  # (ncombs*nfreqs, ntimes)
        self.data = np.hstack(
            [sig.data for sig in self.sigs]
        )  # (ncombs*nfreqs, nsigs*ntimes)
        self.nfreqs, self.ntimes = self.data.shape

    def load_parallel(self, paths):
        print("loading fgs parallel from ", paths)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            print("submitting tasks..")
            future_sigs = [executor.submit(SimSignal, path) for path in paths]
            print("waiting for results..")
            self.sigs = [future.result() for future in future_sigs]
            print("done")
        return self.sigs

