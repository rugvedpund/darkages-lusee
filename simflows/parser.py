import argparse
import sys

DEFAULT_SEED = 42
DEFAULT_NITER = 100000


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--niter", type=int, default=DEFAULT_NITER)
    return parser.parse_args(args)


args = parse_args(sys.argv[1:])
