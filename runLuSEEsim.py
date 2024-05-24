import sys

sys.path.append("/home/rugved/Files/LuSEE/luseepy/simulation/driver/")

from run_sim import SimDriver
import os
import pickle
import simutils

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
dacconfig = simutils.Config(yaml_file).make_da_config()
cmbconfig = simutils.Config(yaml_file).make_cmb_config()

out_dir = os.path.join(os.environ["LUSEE_OUTPUT_DIR"], config.name)
print(f"Output directory: {out_dir}")
if not os.path.exists(out_dir):
    print(f"Creating output directory: {out_dir}")
    os.makedirs(out_dir)


ulsasim = SimDriver(ulsaconfig)
ulsasim.run()

dasim = SimDriver(dacconfig)
dasim.run()

cmbsim = SimDriver(cmbconfig)
cmbsim.run()
