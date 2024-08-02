# %%

import healpy as hp
import lusee
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import simutils

# %%

comblist = [
    "00R",
    "11R",
    "22R",
    "33R",
    "01R",
    "02R",
    "03R",
    "12R",
    "13R",
    "23R",
    "01I",
    "02I",
    "03I",
    "12I",
    "13I",
    "23I",
]


def getbeamcrosspower(comb: str = "00R") -> xr.DataArray:
    lbeam = lusee.Beam("hfss_lbl_3m_75deg.fits")
    assert simutils.is_comb(comb), "Invalid comb"
    angle1, angle2, RorI = comb
    np_realORimag = {"R": np.real, "I": np.imag}[RorI]
    combangles = {"0": 0, "1": -180, "2": -90, "3": -270}  # 0->N,1->S,2->E,3->W
    beam1 = lbeam.rotate(combangles[angle1])
    beam2 = lbeam.rotate(combangles[angle2])
    complex_cross_power = np.array(beam1.power_hp(ellmax=32, Nside=128, cross=beam2))
    cross_power = np_realORimag(complex_cross_power)
    freqs, pixels = np.arange(1, 51), np.arange(hp.nside2npix(128))

    return xr.DataArray(
        cross_power, dims=["freqs", "pixels"], coords={"freqs": freqs, "pixels": pixels}
    )


# %%

for comb in comblist:
    print("doing", comb, "..")
    getbeamcrosspower(comb)

comb_beams = xr.Dataset({comb: getbeamcrosspower(comb) for comb in comblist})
comb_beams = comb_beams.to_array(dim="combs")
comb_beams["combs"] = comb_beams["combs"].astype(str)
comb_beams = comb_beams.to_dataset(name="cross_power")
comb_beams.to_netcdf("netcdf/comb_beams.nc")
comb_beams = xr.open_dataset("netcdf/comb_beams.nc")

# %%

comb_beams = xr.open_dataset("netcdf/comb_beams.nc")
comb_beams["combs"] = comb_beams["combs"].astype(str)
print(comb_beams)
comb_beams.mollview.plot(
    "cross_power", sel=dict(freqs=[1, 10, 20, 30, 40, 50]), col="combs", row="freqs"
)
plt.show()

# %%


# %%
