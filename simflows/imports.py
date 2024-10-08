import argparse
import pickle
import sys

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.animation as animation
import optax

import simflows.jax as simjax
import simflows.load as simloader
import simflows.pca as simpca
import simflows.utils as simutils

# WARN: stupid jax doesn't like float64 by default
jax.config.update("jax_enable_x64", True)
from typing import Dict, List, Tuple

import lusee
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xarray_einstats
from tqdm import tqdm
