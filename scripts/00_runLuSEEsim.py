import sys

sys.path.append("/home/rugved/files/projects/luseepy/simulation/driver/")

import os

import numpy as np
from run_sim import SimDriver

import dalusee.filemanager as filemgr

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

config = filemgr.Config(yaml_file)

ulsaconfig = filemgr.Config(yaml_file).make_ulsa_config()

daconfig = filemgr.Config(yaml_file).make_da_config()

cmbconfig = filemgr.Config(yaml_file).make_cmb_config()

print(f"Output directory: {config._outdir}")
if not os.path.exists(config._outdir):
    print(f"Creating output directory: {config._outdir}")
    os.makedirs(config._outdir)

config.save()

print(ulsaconfig)
ulsasim = SimDriver(ulsaconfig)
ulsasim.run()

print(daconfig)
dasim = SimDriver(daconfig)
dasim.run()

print(cmbconfig)
cmbsim = SimDriver(cmbconfig)
cmbsim.run()
