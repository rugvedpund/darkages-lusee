##--------------------------------------------------------------------##
# file management functions
##--------------------------------------------------------------------##

import os
import xarray as xr
import yaml


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
        self["simulation"]["output"] = (
            f"{self.name}/ulsa.fits"  # $LUSEE_OUTPUT_DIR + self.name + "/ulsa.fits"
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
            f"{self.name}/da.fits"  # $LUSEE_OUTPUT_DIR + self.name + "/da.fits"
        )
        return self

    def make_cmb_config(self):
        print("Making CMB config")
        self["sky"] = {"type": "CMB", "Tcmb": 2.73}
        self["simulation"]["output"] = (
            f"{self.name}/cmb.fits"  # $LUSEE_OUTPUT_DIR + self.name + "/cmb.fits"
        )
        return self

    def save(self):
        filename = os.path.join(self.outdir, self.name)
        filename += ".yaml"
        print("saving config to", filename)
        with open(filename, "w") as file:
            yaml.dump(self, file)


@xr.register_dataset_accessor("configLoader")
class ConfigLoaderAccessor:
    def from_config(self, config: Config):
        import lusee

        ulsapath = config.outdir + "/ulsa.fits"
        dapath = config.outdir + "/da.fits"
        cmbpath = config.outdir + "/cmb.fits"
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
