import argparse
import pickle
import sys
import lusee
import numpy as np
import xarray as xr
import xarray_einstats
import jax
import optax
import jax.numpy as jnp
from typing import Dict, List, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

import dalusee.pca as simpca
import dalusee.load as simloader

# WARN: stupid jax doesn't like float64 by default
jax.config.update("jax_enable_x64", True)
