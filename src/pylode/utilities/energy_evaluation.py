import numpy as np

def periodic_energy_self(potential_func, cell, nneigh=3, avoid_center=True):
    potential = 0
    for ix in range(-nneigh, nneigh+1):
        for iy in range(-nneigh, nneigh+1):
            for iz in range(-nneigh, nneigh+1):
                if ix==0 and iy==0 and iz==0 and avoid_center:
                    continue
                pos_image = ix*cell[0] + iy*cell[1] + iz*cell[2]
                rad = np.linalg.norm(pos_image)
                potential += potential_func(rad)

    return potential

def periodic_energy_pair(potential_func, cell, pos_1, pos_2, nneigh=3):
    potential = 0.
    for ix in range(-nneigh, nneigh+1):
        for iy in range(-nneigh, nneigh+1):
            for iz in range(-nneigh, nneigh+1):
                periodic_shift = ix*cell[0] + iy*cell[1] + iz*cell[2]
                pos_2_image = pos_2 + periodic_shift
                rad = np.linalg.norm(pos_1 - pos_2_image)
                potential += potential_func(rad)

    return potential

def periodic_potential_single(potential_func, frame, nneigh=3, avoid_center=True):
    """
    Compute the energy of an atomic structure, whose atoms interact via 
    a given potential.
    """
    # Initializations
    positions = frame.get_positions()
    natoms = len(frame)
    cell = frame.get_cell()
    potential = 0.

    # Add up the energy contributions from all atom pairs
    # (i,j), where i and j are different atoms, including periodic images
    for i in range(natoms-1):
        for j in range(i+1, natoms):
            pos_1 = positions[i]
            pos_2 = positions[j]
            potential += periodic_energy_pair

    # Add the energy contributions arising from an atom
    # interacting with its own periodic neighbors
    potential += natoms * periodic_energy_self(potential_func, cell, nneigh, avoid_center) 
    return potential


def periodic_potential(potential_func, frames, nneigh=3, avoid_center = True):
    """
    Compute the potential energy for a collection of ase.Atoms frames.
    INPUTS:
    - potential_func: lambda function specifying the interaction potential
    - frames: List of ase.Atoms frames
    - nneigh: Number of periodic neighbors included in the sum along each of
                the cell directions. This means that the total number of 
                neighbors is given by (2*nneigh+1)**3.
    - avoid_center (default: True): If set to False, an atom also interacts
        with itself. The potential evaluated at zero will then be added to
        the total potential of the structure
    
    OUTPUT:
    - np.ndarray of same length as frames containing the potential energies.
    """
    # Initialization
    nframes = len(frames)
    potentials = np.zeros((nframes,))

    # For each frame, evaluate the energy and store the output
    for iframe, frame in enumerate(frames):
        potentials[iframe] = periodic_potential_single(potential_func, frame, nneigh=nneigh, avoid_center=avoid_center)
    
    return potentials