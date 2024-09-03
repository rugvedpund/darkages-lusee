# Load the simulation data into tensors

import os

import xarray as xr

import simflows.utils as simutils

path_here = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path_here, "../")

# NOTE: argparser needs something like: python simtensors.py --config configs/a{0..80..10}_dt3600.yaml
#
# Load the lusee ulsa sim tensor
print("Loading ULSA sim tensor..")
parser = simutils.create_parser()
args = parser.parse_args()
sim = simutils.SimTensor().from_args(args)
print("  saving as netcdf..")
sim.to_netcdf("data/netcdf/sim.nc")

# # WARN: when loading netcdf tensor, ensure string coordniates are loaded as strings
#
# print("loading netcdf tensors..")
# sim = xr.open_dataset("../data/netcdf/sim.nc", chunks={"times": 650})
# sim["combs"] = sim["combs"].astype(str)
# sim = sim.sel(combs=simutils.comblist)
# with open("../data/netcdf/sim_hosvd.nc", "rb") as f:
#     eigs = xr.open_dataset(f)[["eig:combs", "eig:angles"]]
# eigs["combs"] = eigs["combs"].astype(str)
# eigs = eigs.sel(combs=simutils.comblist)
#
# psim = xr.Dataset()
# psim["ulsa"] = sim["ulsa"] @ eigs["eig:combs"] @ eigs["eig:angles"]
# psim["da"] = sim["da"] @ eigs["eig:combs"] @ eigs["eig:angles"]
# psim["cmb"] = sim["cmb"] @ eigs["eig:combs"] @ eigs["eig:angles"]
# data = psim["ulsa"] - psim["ulsa"].mean("times")
