# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 19:16:22 2022

@author: kevin
"""
# Imports specific to this "library"
from projection_coeffs import Density_Projection_Calculator

# Generic imports
import numpy as np
import time
from ase.io import read


def main():
    # Use BaTiO3 data set to show what can be done and provide a feeling
    # for the computational cost
    run_example_BaTiO3()
    
    # Chemical shielding data set
    print('------------------------------------------------------------')
    # run_example_shiftML()


def run_example_BaTiO3():    
    # Get frames and define a dictionary specifying to which index
    # the individual chemical elements are mapped.
    # Since we have 3 elements in this systems, we can access the coefficients
    # using indices 0,1,2, and we map those to the elements O,Ba,Ti.
    # WARNING: There will be errors if the range is not 0,1,...,num_species-1
    frames = read('BaTiO3_Training_set.xyz', ':30')
    species_dict = {'O':0, 'Ti':1, 'Ba':2}
    
    # Define hyperparameters
    hypers = {
        'smearing':2.0, # WARNING: comp. cost scales cubically with 1/smearing
        'max_angular':6,
        'cutoff_radius':4.5,
        'potential_exponent':1, # currently, only the exponent p=1 is supported
        'compute_gradients':True       
    }
    
    # Evaluate the features on all frames and record required time
    tstart = time.time()
    calculator = Density_Projection_Calculator(**hypers)
    calculator.transform(frames, species_dict)
    dt = time.time() - tstart
    
    # Example for how to get features and gradients
    # Check out get_features() and get_feature_gradients() for a detailed
    # description of the array format
    features = calculator.get_features()
    gradients = calculator.get_feature_gradients()
    print('BaTiO3 data set')
    print('Shape of obtained feature array = ', features.shape)
    print('Shape of obtained gradient array = ', gradients.shape)
    
    # Reference values to understand the array shapes
    print('\nValues for reference to understand the array shapes:')
    print('Number of frames =', len(frames))
    print('Total number of environments = ', sum([len(frame) for frame in frames]))
    print('Number of chemical species = ', len(species_dict))
    print('nmax = ', 1)
    print('lmax = ', hypers['max_angular'])
    print('(lmax+1)^2 = ', (hypers['max_angular']+1)**2)
    print('Note: nmax=1 by default for 1/r LODE')
    
    # Estimate required time to compute features for all frames
    print('\nComputational cost:')
    print(f'Required time for {len(frames)} frames = {dt:4.1f}s')   
    frames_all = read('BaTiO3_Training_set.xyz', ':')
    N_all = len(frames_all)
    dt_all = N_all/len(frames)*dt
    print(f'Estimated time for {N_all} structures {dt_all:4.1f}s = {dt_all/60.:4.1f}min')



def run_example_shiftML():
    frames = read('shiftml.xyz',':')
    species_dict = {'C':0, 'H':1, 'N':2, 'O':3}
    
    # Define hyperparameters
    hypers = {
        'smearing':3.,
        'max_angular':2,
        'cutoff_radius':3.5,
        'potential_exponent':1,
        'compute_gradients':True       
    }
    num_strucs = len(frames)
    num_atoms = np.array([len(frame) for frame in frames])
    num_env = np.sum(num_atoms)
    print('\nShiftML data set')
    print('Number of structures         = ', num_strucs)
    print('Total # of environments      = ', num_env)
    print('Average # of atoms per frame = ', np.round(num_env / len(frames),1))
    
    frames = frames[:20]
    print(f'Number of atoms of first {len(frames)} structures = ', [len(frame) for frame in frames])
    
    tstart = time.time()
    calculator = Density_Projection_Calculator(**hypers)
    calculator.transform(frames, species_dict)
    features = calculator.get_features()
    gradients = calculator.get_feature_gradients()
    tend = time.time()
    dt = tend - tstart
    print(f'Required time for first {len(frames)} frames = {np.round(dt,2)}s')
    print(f'Estimated time for all frames = {np.round(dt/len(frames)*num_strucs/60,2)}min')


# Note for computational cost:
# version 1: BaTiO3: estimated total 24min, ShiftML: 62s for first 20 frames
# version 2: Precompute as much as possible using numpy arrays -> 18min, 51s
#            Some tests show that main loop is indeed the main contribution
#            to computational cost + fixed 1s contribution from splining in
#            the very beginning.
# version 3: 



if __name__ == '__main__':
    main()