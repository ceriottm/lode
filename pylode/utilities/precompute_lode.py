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

from pylode import Density_Projection_Calculator as LODE

def lode_get_features(frames, show_progress=False, **hypers):
    """Calculate LODE feature array from an atomic dataset.
    
    Asssuming a constant number of atoms in each set.
    
    Parameters
    ----------
    frames : list[ase.Atoms]
        List of datasets for calculating features
    show_progress : bool
        Show progress bar
    hypers : kwargs
        Kwargs of hyperparameters. 
        See pylode.Density_Projection_Calculator for details.
        
    Returns
    -------
    X : np.ndarray of shape (n_sets, n_atoms, n_chem_species, max_radial, (max_angular + 1)**2)
        feature array
    """
    n_frames = len(frames)
    n_atoms = len(frames[0])
    
    species_dict = {}
    for frame in frames:
        # Move atoms in unitcell
        frame.wrap()
        #Get atomic species in dataset
        species_dict.update({atom.symbol: atom.number for atom in frame})

    calculator = LODE(**hypers)

    calculator.transform(frames=frames,
                         species_dict=species_dict,
                         show_progress=show_progress)
    


    X = calculator.get_features()
    # reshape lode features
    X = X.reshape(n_frames, n_atoms, *X.shape[1:])
    
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
    parser.add_argument("-b",
                        dest="radial_basis",
                        type=str,
                        default="monomial",
                        const="monomial",
                        nargs='?',
                        choices=["monomial", "GTO"],
                        help="The radial basis. Currently "
                        "implemented are 'GTO' and 'monomial'.")
    parser.add_argument("-e",
                        dest="potential_exponent",
                        type=int,
                        default=1,
                        const=1,
                        nargs='?',
                        choices=[0, 1],
                        help="potential exponent: "
                        "p=0 uses Gaussian densities, "
                        "p=1 is LODE using 1/r (Coulomb) densities")
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
    np.save(args.outfile, lode_get_features(frames,
                                            show_progress=True,
                                            **args.__dict__))

if __name__=="__main__":
    main()
