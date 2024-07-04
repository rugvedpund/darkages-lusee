import sys

sys.path.append("/home/rugved/Files/LuSEE/luseepy/simulation/driver/")

from run_sim import SimDriver
import os
import pickle
import simflows
import numpy as np

##---------------------------------------------------------------------------##
"""
This script is used to run the simulation using the configuration file.
1. yaml file is passed as a command line argument, default luseepy format for ulsa map
2. loaded into simutils.Config object. create corresponding config objects for ulsa, da, cmb
3. run all three simulations using the config, SimDriver object
4. save all three fits outputs into a new dir in outputs/
"""
##---------------------------------------------------------------------------##

if len(sys.argv) < 2:
    print("Usage: python runLuSIM.py <config_file>")
    sys.exit(1)
yaml_file = sys.argv[1]

config = simutils.Config(yaml_file)

ulsaconfig = simutils.Config(yaml_file).make_ulsa_config()

daconfig = simutils.Config(yaml_file).make_da_config()

cmbconfig = simutils.Config(yaml_file).make_cmb_config()

print(f"Output directory: {config.outdir}")
if not os.path.exists(config.outdir):
    print(f"Creating output directory: {config.outdir}")
    os.makedirs(config.outdir)

print(ulsaconfig)
ulsasim = SimDriver(ulsaconfig)
ulsasim.run()

print(daconfig)
dasim = SimDriver(daconfig)
dasim.run()

print(cmbconfig)
cmbsim = SimDriver(cmbconfig)
cmbsim.run()

# amp = np.logspace(np.log10(0.01),np.log10(100),10)
# for a in amp:
#     config = simutils.Config(yaml_file)
#     cfg_name = f"amp{a:1.1e}"

#     ulsaconfig = simutils.Config(yaml_file).make_ulsa_config()
#     ulsaconfig["simulation"]["output"] = f"{cfg_name}/ulsa.fits"

#     daconfig = simutils.Config(yaml_file).make_da_config()
#     daconfig["sky"]["A"] *= float(a)
#     daconfig["simulation"]["output"] = f"{cfg_name}/da.fits"

#     cmbconfig = simutils.Config(yaml_file).make_cmb_config()
#     cmbconfig["sky"]["Tcmb"] *= float(a)
#     cmbconfig["simulation"]["output"] = f"{cfg_name}/cmb.fits"

#     out_dir = os.path.join(os.environ["LUSEE_OUTPUT_DIR"], cfg_name)
#     print(f"Output directory: {out_dir}")
#     if not os.path.exists(out_dir):
#         print(f"Creating output directory: {out_dir}")
#         os.makedirs(out_dir)

#     print(ulsaconfig)
#     ulsasim = SimDriver(ulsaconfig)
#     ulsasim.run()

#     print(daconfig)
#     dasim = SimDriver(daconfig)
#     dasim.run()

#     print(cmbconfig)
#     cmbsim = SimDriver(cmbconfig)
#     cmbsim.run()
