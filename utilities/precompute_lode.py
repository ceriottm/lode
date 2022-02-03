#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precompute LODE features for given dataset.

Calculations may take along time and memory. Be careful!
"""

import sys

sys.path.append("../")

import numpy as np

from ase.io import read
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from pylode.projection_coeffs import Density_Projection_Calculator as LODE

################
#### INPUT #####
################

input_file = "../datasets/point_charges_Training_set.xyz"
hypers_lode = dict(
    max_angular=6,
    cutoff_radius=3,
    potential_exponent=1,  # currently, only the exponent p=1 is supported
    compute_gradients=False)
r_smearing = [1, 0.5]  # computational cost scales cubically with 1/smearing
species_dict = {'Na': 0, 'Cl': 1}

######################
#### COMPUTATION #####
######################

frames = read(input_file, index=':10')
n_frames = len(frames)

# Move atoms in unitcell
for frame in frames:
    frame.wrap()

f_train = 0.75  # factor of the train set picked from the total set

f_test = 1 - f_train
i_train = train_test_split(np.arange(n_frames),
                           test_size=f_test,
                           random_state=0)[0]

frames_train = [frames[i] for i in i_train]

for smearing in tqdm(r_smearing, leave=True, desc="Precompute features"):
    fname_precomputed = f"../datasets/precomputed_lode_{smearing}"

    hypers_lode["smearing"] = smearing

    calculator = LODE(**hypers_lode)
    lode_rep = calculator.transform(frames, species_dict)
    X_raw = lode_rep.get_features(calculator)

    np.save(fname_precomputed, X_raw)
