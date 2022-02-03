#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precompute LODE features for given dataset.

Calculations may take along time and memory. Be careful!
"""
import argparse
import sys

sys.path.append("../")

import numpy as np

from ase.io import read

from pylode.projection_coeffs import Density_Projection_Calculator as LODE

HYPERS_LODE = dict(
    max_angular=2,
    cutoff_radius=3,
    potential_exponent=1,  # currently, only the exponent p=1 is supported
    compute_gradients=False)

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f',
                    dest='input_file',
                    type=str,
                    help="Trajectory for constructing features.",
                    default="../datasets/point_charges_Training_set.xyz")
parser.add_argument('-i',
                    dest='index',
                    type=str,
                    help="slicing string for trjectory slicing",
                    default=":")
parser.add_argument('-s',
                    dest='smearing',
                    type=float,
                    help="Smearing of the Gaussain (Å)."
                    "Note that computational cost scales "
                    "cubically with 1/smearing.",
                    default=1)
parser.add_argument('-o',
                    dest='output',
                    type=str,
                    help="Output filename for the feature matrix.",
                    default="precomputed_lode")

args = parser.parse_args()
frames = read(args.input_file, index=args.index)

# Get atomic species in dataset
global_species = set()
for frame in frames:
    global_species.update(frame.get_chemical_symbols())
species_dict = {k: i for i, k in enumerate(global_species)}

# Move atoms in unitcell
for frame in frames:
    frame.wrap()

HYPERS_LODE["smearing"] = args.smearing

calculator = LODE(**HYPERS_LODE)
calculator.transform(frames, species_dict)

np.save(args.output, calculator.get_features())
