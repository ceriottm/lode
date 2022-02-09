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

def lode_get_features(frames, **hypers):
    """Calculate LODE feature array from an atomic dataset.
    
    Asssuming a constant number of atoms in each set.
    
    Parameters
    ----------
    frames : list[ase.Atoms]
        List of datasets for calculating features
    hypers : kwargs
        Kwargs of hyperparameters. 
        See pylode.Density_Projection_Calculator for details.
        
    Returns
    -------
    X : np.ndarray of shape (n_sets, n_atoms, n_)
        feature array
    """
    n_frames = len(frames)
    n_atoms = len(frames[0])

    # Get atomic species in dataset
    global_species = set()
    for frame in frames:
        global_species.update(frame.get_chemical_symbols())
    species_dict = {k: i for i, k in enumerate(global_species)}

    # Move atoms in unitcell
    for frame in frames:
        frame.wrap()

    calculator = LODE(**hypers)
    calculator.transform(frames, species_dict)

    X = calculator.get_features()
    # reshape lode features in the shape (n_sets, n_atoms, n_features)
    X = X.reshape(n_frames, n_atoms, np.prod(X.shape[1:]))
    
    return X


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f",
                        dest="input_file",
                        type=str,
                        help="Trajectory for constructing features.",
                        default="../datasets/point_charges_Training_set.xyz")
    parser.add_argument("-i",
                        dest="index",
                        type=str,
                        help="slicing string for trjectory slicing",
                        default=":")
    parser.add_argument("-n",
                        dest="max_radial",
                        type=int,
                        help="Number of radial functions",
                        default=1)
    parser.add_argument("-l",
                        dest="max_angular",
                        type=int,
                        help="Number of angular functions",
                        default=6)
    parser.add_argument("-rc",
                        dest="cutoff_radius",
                        type=float,
                        help="Environment cutoff (Å)",
                        default=4.0)
    parser.add_argument("-s",
                        dest="smearing",
                        type=float,
                        help="Smearing of the Gaussain (Å)."
                        "Note that computational cost scales "
                        "cubically with 1/smearing.",
                        default=1)
    parser.add_argument('-g',
                        dest='compute_gradients',
                        action='store_true',
                        help="Compute gradients")
    parser.add_argument("-o",
                        dest="outfile",
                        type=str,
                        help="Output filename for the feature matrix.",
                        default="precomputed_lode")

    args = parser.parse_args()
    frames = read(args.input_file, index=args.index)

    hypers_lode = dict(
        potential_exponent=1,  # currently, only the exponent p=1 is supported
        max_radial=args.max_radial,
        max_angular=args.max_angular,
        cutoff_radius=args.cutoff_radius,
        compute_gradients=args.compute_gradients,
        smearing=args.smearing)

    np.save(args.outfile, lode_get_features(frames, **hypers_lode))


if __name__=="__main__":
    main()
