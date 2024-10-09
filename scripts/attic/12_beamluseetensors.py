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
lbeam = lusee.Beam("hfss_lbl_3m_75deg.fits")


def getluseebeam(angle_direction: str = "N") -> xr.DataArray:
    combangles = {"N": 0, "S": -180, "E": -90, "W": -270}  # 0->N,1->S,2->E,3->W
    beam = lbeam.rotate(combangles[angle_direction])
    power = np.array(beam.power_hp(ellmax=32, Nside=128))
    freqs, pixels = np.arange(1, 51), np.arange(hp.nside2npix(64))
    return xr.DataArray(
        power, dims=["freqs", "pixels"], coords={"freqs": freqs, "pixels": pixels}
    )


def getbeamcrosspower(comb: str = "00R", common_angle: np.float64 = 0) -> xr.DataArray:
    print("doing", comb, "for common angle", common_angle, "..")
    assert simutils.is_comb(comb), "Invalid comb"
    angle1, angle2, RorI = comb
    np_realORimag = {"R": np.real, "I": np.imag}[RorI]
    combangles = {"0": 0, "1": -180, "2": -90, "3": -270}  # 0->N,1->S,2->E,3->W
    print("  rotating..")
    beam1 = lbeam.rotate(common_angle + combangles[angle1])
    beam2 = lbeam.rotate(common_angle + combangles[angle2])
    print("  computing power..")
    complex_cross_power = np.array(beam1.power_hp(ellmax=32, Nside=64, cross=beam2))
    cross_power = np_realORimag(complex_cross_power)
    freqs, pixels = np.arange(1, 51), np.arange(hp.nside2npix(64))
    print("  returning xarray..")

    return xr.DataArray(
        cross_power, dims=["freqs", "pixels"], coords={"freqs": freqs, "pixels": pixels}
    )


# %%
##--------------------------------------------------------------------##

# comb_beams = xr.Dataset({comb: getbeamcrosspower(comb) for comb in comblist})
# comb_beams = comb_beams.to_array(dim="combs")
# comb_beams["combs"] = comb_beams["combs"].astype(str)
# comb_beams = comb_beams.to_dataset(name="cross_power")
# comb_beams.to_netcdf("netcdf/comb_beams.nc")

# %%

# combangle_beams = xr.Dataset()
# for cba in [0, 10, 20, 30, 40, 50, 60, 70, 80]:
#     print("doing", cba, "degrees..")
#     combangle_beams[cba] = xr.Dataset(
#         {comb: getbeamcrosspower(comb, cba) for comb in comblist}
#     ).to_array(dim="combs")
# combangle_beams = combangle_beams.to_array(dim="angles")
# combangle_beams["combs"] = combangle_beams["combs"].astype(str)
# combangle_beams = combangle_beams.to_dataset(name="cross_power")
# combangle_beams.to_netcdf("netcdf/combangle_beams.nc")

# %%
##--------------------------------------------------------------------##


combangle_beams = xr.open_dataset("netcdf/combangle_beams.nc")
combangle_beams["combs"] = combangle_beams["combs"].astype(str)
print(combangle_beams)

# %%

comb_beams = xr.open_dataset("netcdf/comb_beams.nc")
comb_beams["combs"] = comb_beams["combs"].astype(str)
print(comb_beams)

g = combangle_beams.mollview.plot(
    "cross_power",
    sel=dict(
        freqs=10,
        angles=[0, 10, 20, 30, 40, 50, 60, 70, 80],
        combs=["00R", "11R", "22R", "33R"],
    ),
    col="angles",
    row="combs",
)
plt.savefig("tex/figures/combangle_beams_auto.pdf")
# plt.show()

g = combangle_beams.mollview.plot(
    "cross_power",
    sel=dict(
        freqs=10,
        angles=[0, 10, 20, 30, 40, 50, 60, 70, 80],
        combs=["01R", "02R", "03R", "12R", "13R", "23R"],
    ),
    col="angles",
    row="combs",
)
plt.savefig("tex/figures/combangle_beams_crossR.pdf")
# plt.show()

g = combangle_beams.mollview.plot(
    "cross_power",
    sel=dict(
        freqs=10,
        angles=[0, 10, 20, 30, 40, 50, 60, 70, 80],
        combs=["01I", "02I", "03I", "12I", "13I", "23I"],
    ),
    col="angles",
    row="combs",
)
plt.savefig("tex/figures/combangle_beams_crossI.pdf")
# plt.show()

# %%

g = comb_beams.mollview.plot(
    "cross_power",
    sel=dict(freqs=[1, 10, 20, 30, 40], combs=["00R", "11R", "22R", "33R"]),
    row="combs",
    col="freqs",
)
plt.savefig("tex/figures/comb_beams_auto.pdf")
plt.show()

g = comb_beams.mollview.plot(
    "cross_power",
    sel=dict(
        freqs=[1, 10, 20, 30, 40], combs=["01R", "02R", "03R", "12R", "13R", "23R"]
    ),
    row="combs",
    col="freqs",
)
plt.savefig("tex/figures/comb_beams_crossR.pdf")
plt.show()

g = comb_beams.mollview.plot(
    "cross_power",
    sel=dict(
        freqs=[1, 10, 20, 30, 40], combs=["01I", "02I", "03I", "12I", "13I", "23I"]
    ),
    row="combs",
    col="freqs",
)
plt.savefig("tex/figures/comb_beams_crossI.pdf")
plt.show()


# %%

lusee_beam = xr.Dataset()
for direction in ["N", "S", "E", "W"]:
    lusee_beam[direction] = getluseebeam(direction)
lusee_beam = lusee_beam.to_array(dim="direction")
lusee_beam["direction"] = lusee_beam["direction"].astype(str)
lusee_beam = lusee_beam.to_dataset(name="power")
lusee_beam.to_netcdf("netcdf/lusee_beam.nc")
lusee_beam = xr.open_dataset("netcdf/lusee_beam.nc")

# %%

lusee_beam = xr.open_dataset("netcdf/lusee_beam.nc")
lusee_beam["direction"] = lusee_beam["direction"].astype(str)
print(lusee_beam)

# %%

g = lusee_beam.mollview.plot(
    "power",
    sel=dict(freqs=[1, 10, 20, 30, 40], direction=["N", "S", "E", "W"]),
    row="direction",
    col="freqs",
)
plt.savefig("tex/figures/lusee_beams.pdf")
plt.show()

# %%

from loadtensors import tensor

# %%

eigs = tensor[["eig:combs", "eig:angles"]]

proj_baselines = xr.Dataset()
proj_baselines["p_ac"] = (
    combangle_beams["cross_power"] @ eigs["eig:combs"] @ eigs["eig:angles"]
)
proj_baselines["p_a"] = combangle_beams["cross_power"] @ eigs["eig:angles"]
proj_baselines["p_c"] = combangle_beams["cross_power"] @ eigs["eig:combs"]
proj_baselines.to_netcdf("netcdf/proj_baselines.nc")
proj_baselines = xr.open_dataset("netcdf/proj_baselines.nc")
print(proj_baselines)

# %%

g = proj_baselines.mollview.plot(
    "p_ac",
    sel=dict(freqs=40),
    col="angles2",
    row="combs2",
)
plt.savefig("tex/figures/p_ac_40.pdf")
# plt.show()

g = proj_baselines.mollview.plot(
    "p_c",
    sel=dict(freqs=40),
    col="angles",
    row="combs2",
)
plt.savefig("tex/figures/p_c_40.pdf")
# plt.show()

g = proj_baselines.mollview.plot(
    "p_a",
    sel=dict(freqs=40),
    col="angles2",
    row="combs",
)
plt.savefig("tex/figures/p_a_40.pdf")
# plt.show()

# %%


# %%
