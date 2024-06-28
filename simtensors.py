import lusee
import simutils
import simflows as sf
import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
import xarray_einstats as xstats
from xarray_einstats import linalg
import tensorly as tl

log = mpl.colors.LogNorm()
symlog = mpl.colors.SymLogNorm(linthresh=1e2)

##---------------------------------------------------------------------------##


class BaseSimTensor:
    def from_luseeData(self, luseeData: lusee.Data, attrs: dict = {}):
        self.luseeData = luseeData  # (times, combs, freqs)
        assert self.luseeData.data.shape[1] == 16
        self.freqs = self.luseeData.freq
        self.times = np.arange(self.luseeData.Ntimes)
        self.combs = simutils.combs(nbeams=4)
        self.tensor = xr.DataArray(
            self.luseeData.data,
            coords={
                "times": self.times,
                "combs": self.combs,
                "freqs": self.freqs,
            },
            dims=["times", "combs", "freqs"],
            attrs=attrs,
        )
        return self

    def __repr__(self):
        return f"<BaseSimTensor>\n{repr(self.tensor)}"


class SimTensor:
    def from_config(self, config: sf.Config):
        ulsapath = config.outdir + "/ulsa.fits"
        dapath = config.outdir + "/da.fits"
        cmbpath = config.outdir + "/cmb.fits"
        self.ulsa = BaseSimTensor().from_luseeData(
            lusee.Data(ulsapath), attrs={"type": "ulsa"}
        )
        self.da = BaseSimTensor().from_luseeData(
            lusee.Data(dapath), attrs={"type": "da"}
        )
        self.cmb = BaseSimTensor().from_luseeData(
            lusee.Data(cmbpath), attrs={"type": "cmb"}
        )
        self.tensor = xr.Dataset(
            data_vars={
                "ulsa": self.ulsa.tensor,
                "da": self.da.tensor,
                "cmb": self.cmb.tensor,
            },
        )
        return self


class ExtendedSimTensor:
    def from_args(self, args):
        configpaths = args.configs
        datasets = []
        angles = []
        for path in configpaths:
            config = sf.Config(path)
            angles.append(config["beam_config"]["common_beam_angle"])
            simtensor = SimTensor().from_config(config).tensor
            datasets.append(simtensor)
        self.datasets = datasets
        self.angles = xr.DataArray(angles, dims="angles")
        self.tensor = xr.concat(self.datasets, dim=self.angles)
        return self


# (angles, times, combs, freqs)

##---------------------------------------------------------------------------##
if __name__ == "__main__":
    parser = simutils.create_parser()
    args = parser.parse_args()

    sky = ExtendedSimTensor().from_args(args)    # (angles, times, combs, freqs)
    ulsa = sky.tensor.ulsa - sky.tensor.ulsa.mean(
        dim="times"
    )  # (angles, times, combs, freqs)
    D = tl.tensor(ulsa.data.copy())
    D0 = tl.unfold(D, mode=0)  # (angles, times*combs*freqs)
    D1 = tl.unfold(D, mode=1)  # (times, combs*freqs*angles)
    D2 = tl.unfold(D, mode=2)  # (combs, freqs*angles*times)
    D3 = tl.unfold(D, mode=3)  # (freqs, angles*times*combs)

    # (angles, times, combs, freqs)
    print("unfolding and doing higher order SVD..")
    U0, S0, _ = scipy.linalg.svd(D0, full_matrices=False)  # angles
    U1, S1, _ = scipy.linalg.svd(D1, full_matrices=False)  # times
    U2, S2, _ = scipy.linalg.svd(D2, full_matrices=False)  # combs
    U3, S3, _ = scipy.linalg.svd(D3, full_matrices=False)  # freqs
    Z = tl.tenalg.multi_mode_dot(D, [U0.T, U1.T, U2.T, U3.T], modes=[0, 1, 2, 3])

    eigenall = xr.DataArray(
        tl.tenalg.multi_mode_dot(Z, [U3], modes=[3]), coords=ulsa.coords, dims=ulsa.dims
    )
    eigenangles = xr.DataArray(
        tl.tenalg.multi_mode_dot(Z, [U0, U3], modes=[0, 3]),
        coords=ulsa.coords,
        dims=ulsa.dims,
    )
    eigencombs = xr.DataArray(
        tl.tenalg.multi_mode_dot(Z, [U2, U3], modes=[2, 3]),
        coords=ulsa.coords,
        dims=ulsa.dims,
    )

    symlog = mpl.colors.SymLogNorm(linthresh=1e-5)
    eigenall.sel(angles=[0, 20, 40, 60, 80], combs=["00R", "11R", "22R", "33R"]).plot(
        col="combs", row="angles", x="times", norm=symlog
    )
    plt.show()

    eigenangles.sel(angles=[0, 20, 40, 60, 80], combs=["00R", "11R", "22R", "33R"]).plot(
        col="combs", row="angles", x="times", norm=symlog
    )
    plt.show()

    eigencombs.sel(angles=[0, 20, 40, 60, 80], combs=["00R", "11R", "22R", "33R"]).plot(
        col="combs", row="angles", x="times", norm=symlog
    )
    plt.show()

    import ipdb; ipdb.set_trace()

    # # plot single waterfall
    # sky.tensor.ulsa.sel(combs="00R", angles=0).plot(norm=log)
    # plt.show()

    # # plot all combs for a single angle
    # sky.tensor.ulsa.sel(angles=0).plot(
    #     x="times", y="freqs", col="combs", col_wrap=4, norm=symlog, cmap="viridis"
    # )
    # plt.show()

    # # pcolormesh plot all angles for a single comb, avgd over freqs
    # sky.tensor.ulsa.sel(angles=[0,20,40,60,80],combs="00R").mean(dim="freqs").plot(x="times",norm=log)
    # plt.show()

    # # line plot all angles for auto combs, avgd over freqs
    # sky.tensor.ulsa.sel(angles=[0,20,40,60,80],combs=["00R","11R","22R","33R"]).mean(dim="freqs").plot.line(x="times",hue="angles",col="combs",col_wrap=4)
    # plt.show()

    # # line plot all angles for all combs, avgd over times
    # sky.tensor.ulsa.sel(angles=[0,20,40,60,80]).mean(dim="times").plot.line(x="freqs",hue="angles",col="combs",col_wrap=4)
    # plt.show()

    # # plot single waterfall
    # Z.sel(combs="00R",angles=0).plot(norm=symlog)
    # plt.show()

    # # plot all combs for a single angle
    # Z.sel(angles=0).plot(
    #     x="times", y="freqs", col="combs", col_wrap=4, norm=symlog, cmap="bwr"
    # )
    # plt.show()

    # plot all angles for a single comb, avgd over freqs
    Z.sel(angles=[0, 20, 40, 60, 80], combs="00R").mean(dim="freqs").plot(
        x="times", norm=symlog
    )
    plt.show()

    import ipdb

    ipdb.set_trace()

    # print("doing tucker..")
    # (core, (Uf,Ut,Uc,Ua)), err = tl.decomposition.tucker(
    #     D, rank=None, return_errors=True, verbose=True
    # )
