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
from sklearn.model_selection import train_test_split

from pylode.projection_coeffs import Density_Projection_Calculator as LODE

HYPERS_LODE = dict(
    max_angular=6,
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
parser.add_argument('-r',
                    dest='f_train',
                    type=float,
                    help="Factor of the train set picked from the total set",
                    default=0.75)
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

# Get frames for test set
i_train = train_test_split(np.arange(len(frames)),
                           train_size=args.f_train,
                           random_state=0)[0]

frames_train = [frames[i] for i in i_train]

HYPERS_LODE["smearing"] = args.smearing

calculator = LODE(**HYPERS_LODE)
lode_rep = calculator.transform(frames, species_dict)
X_raw = lode_rep.get_features(calculator)

np.save(args.output, X_raw)
