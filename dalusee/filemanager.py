##--------------------------------------------------------------------##
# file management functions
##--------------------------------------------------------------------##

import os
import xarray as xr
import yaml


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", help="config yaml paths")
    parser.add_argument("--outputs", nargs="*", help="output dir path")
    parser.add_argument("--params_yaml", type=str, help="param yaml path")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--comb", type=str, help="e.g. 00R, auto, all", default="00R")
    return parser.parse_args(args)


def check_file_exists(file_path):
    return os.path.exists(file_path)


class Config(dict):
    def __init__(self, yaml_file):
        print("Loading config from ", yaml_file)
        self.yaml_file = yaml_file
        self.from_yaml(self.yaml_file)

    def __repr__(self):
        return yaml.dump(self)

    def __str__(self):
        return yaml.dump(self)

    @property
    def _name(self):
        yamlnamedotyaml = os.path.basename(self.yaml_file)
        yamlname = os.path.splitext(yamlnamedotyaml)[0]
        return yamlname

    @property
    def _outdir(self):
        return os.path.join(os.environ["LUSEE_OUTPUT_DIR"], self._name)

    def _outname(self):
        # TODO: add da/cmb/ulsa to the yaml name
        return self._name

    def save(self):
        filename = os.path.join(self._outdir, self._name)
        filename += ".yaml"
        print("saving config to", filename)
        with open(filename, "w") as file:
            yaml.dump(self, file)

    def from_yaml(self, yaml_file):
        with open(yaml_file) as f:
            yml = yaml.safe_load(f)
        self.update(yml)
        return self

    def make_ulsa_config(self):
        print("Making ULSA config")
        self["sky"] = {"file": "ULSA_32_ddi_smooth.fits"}
        self["simulation"]["output"] = (
            f"{self._name}/ulsa.fits"  # $LUSEE_OUTPUT_DIR + self.name + "/ulsa.fits"
        )
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
        self["simulation"]["output"] = (
            f"{self._name}/da.fits"  # $LUSEE_OUTPUT_DIR + self.name + "/da.fits"
        )
        return self

    def make_cmb_config(self):
        print("Making CMB config")
        self["sky"] = {"type": "CMB", "Tcmb": 2.73}
        self["simulation"]["output"] = (
            f"{self._name}/cmb.fits"  # $LUSEE_OUTPUT_DIR + self.name + "/cmb.fits"
        )
        return self


@xr.register_dataset_accessor("configLoader")
class ConfigLoaderAccessor:
    def from_config(self, config: Config):
        import lusee

        ulsapath = config._outdir + "/ulsa.fits"
        dapath = config._outdir + "/da.fits"
        cmbpath = config._outdir + "/cmb.fits"
        self.ulsa = xr.DataArray().luseeDataLoader.from_luseeData(
            lusee.Data(ulsapath), name="ulsa"
        )
        self.da = xr.DataArray().luseeDataLoader.from_luseeData(
            lusee.Data(dapath), name="da"
        )
        self.cmb = xr.DataArray().luseeDataLoader.from_luseeData(
            lusee.Data(cmbpath), name="cmb"
        )
        self._obj = xr.Dataset(
            data_vars={"ulsa": self.ulsa, "da": self.da, "cmb": self.cmb}
        )
        return self._obj

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
