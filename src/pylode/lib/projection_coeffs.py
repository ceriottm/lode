# -*- coding: utf-8 -*-
"""

Python version of the LODE implementation.
Currently, only the exponent p=1 with the optimal radial basis r^l for each
angular channel l=0,1,2,...,lmax is supported.

"""

# Generic imports
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = (lambda i, **kwargs: i)

from .kvec_generator import KvectorGenerator
from .radial_basis import RadialBasis
from .spherical_harmonics import evaluate_spherical_harmonics


class DensityProjectionCalculator():
    def __init__(self,
                 max_radial,
                 max_angular,
                 cutoff_radius,
                 smearing,
                 radial_basis,
                 compute_gradients=False,
                 potential_exponent=1,
                 subtract_self=False):
        """
        Compute the spherical expansion coefficients.

        Initialize the calculator using the hyperparameters.
        All the needed splines that only depend on the hyperparameters
        are prepared as well by storing the values.

        Parameters
        ----------
        max_radial : int
            Number of radial functions
        max_angular : int
            Number of angular functions
        cutoff_radius : float
            Environment cutoff (Å)
        smearing : float
            Smearing of the Gaussain (Å). Note that computational cost scales
            cubically with 1/smearing.
        radial_basis : str
            The radial basis. Currently implemented are 'GTO' and 'monomial'.
            For monomial: Only use one radial basis r^l for each angular
            channel l leading to a total of (lmax+1)^2 features.
        compute_gradients : bool
            Compute gradients
        potential_exponent : int
            potential exponent: p=0 uses Gaussian densities,
            p=1 is LODE using 1/r (Coulomb) densities"
        subtract_self : bool
            Subtract contribution from the central atom.

        Attributes
        ----------
        features : array
            the computed projection coefficients in the format:
            The projection coefficients as an array of dimension:
                num_environ x num_chem_species x num_radial x num_lm_coefficients,
            where:
                num_environ = total number of atoms in the system summed over
                                all frames
                num_chem_species = number of chemical species
                num_radial = nmax
                num_lm_coefficients = (lmax+1)^2
        feature_gradients : array
            the gradients of the projection coefficients
            The returned array has dimensions:
            num_environm_squared x 3 x num_chemical_species x num_radial x num_lm_coefficients,

            The factor of 3 corresponds to x,y,z-components.
            Otherwise, the specification is almost identical to get_features, except
            that the first axis specifying the atomic environment now runs over
            all pairs (i,j).
            Example: For a structure containing 3 atoms labeled as (0,1,2),
            the gradients are stored as
            gradients[0] = dV_0/dr_0
            gradients[1] = dV_0/dr_1
            gradients[2] = dV_0/dr_2
            gradients[3] = dV_1/dr_0
            gradients[4] = dV_1/dr_1
            ...
            gradients[8] = dV_2/dr_2

            If multiple frames are present, all these coefficients are
            concatenated along the 0-th axis, as usual e.g. for SOAP vectors
            in librascal.

        representation_info : array
            Stuff for interacting to interact with atomistic-ml-storage. 
        """
        self.max_radial = max_radial
        self.max_angular = max_angular
        self.cutoff_radius = cutoff_radius
        self.radial_basis = radial_basis.lower()
        self.smearing = smearing
        self.potential_exponent = potential_exponent
        self.compute_gradients = compute_gradients
        self.subtract_self = subtract_self

        if self.potential_exponent not in [0, 1]:
            raise ValueError("Potential exponent has to be one of 0 or 1!")

        if self.radial_basis not in ["monomial", "gto"]:
            raise ValueError(f"{self.radial_basis} is not an implemented"
                              " basis. Try 'monomial' or 'GTO'.")

        if self.radial_basis == "monomial" and self.max_radial != 1:
            raise ValueError("For monomial basis only `max_radial=1` "
                             "is allowed.")

        self.num_features_bare = self.max_radial * (self.max_angular + 1)**2
        self.radial_proj = RadialBasis(self.max_radial, self.max_angular,
                                       self.cutoff_radius, self.smearing,
                                       self.radial_basis)

        self.radial_proj.precompute_radial_projections(np.pi/self.smearing)

    def transform(self, frames, species_dict, show_progress=False):
        """
        Computes the features and (if compute_gradients == True) gradients
        for all the provided frames. The features and gradients can be
        obtained using get_features() and get_feature_gradients().

        Parameters
        ----------
        frames : ase.Atoms
            List containing all ase.Atoms structures

        species_dict: Dictionary
            All chemical species present in the system with their mapping to
            indices, e.g. for BaTiO3: {'O':0, 'Ti':1, 'Ba':2}
        show_progress : bool
            Show progress bar for frame analysis

        Returns
        -------
        None, but stores the projection coefficients and (if desired)
        gradients as arrays as `features` and `features_gradients`.
        """
        # Define variables determining size of feature vector coming from frames
        num_atoms_per_frame = np.array([len(frame) for frame in frames])
        num_atoms_total = np.sum(num_atoms_per_frame)
        num_chem_species = len(species_dict)

        # Initialize arrays in which to store all features
        self.features = np.zeros((num_atoms_total, num_chem_species,
                                  self.max_radial, (self.max_angular+1)**2))
        if self.compute_gradients:
            num_gradients = np.sum(num_atoms_per_frame**2)
            self.feature_gradients = np.zeros((num_gradients, 3, num_chem_species,
                                               self.max_radial, (self.max_angular+1)**2))

        # For each frame, compute the projection coefficients
        current_index = 0
        gradient_index = 0

        if show_progress:
            frame_generator = tqdm(frames)
        else:
            frame_generator = frames

        self.representation_info = np.zeros([len(frames) * np.sum(num_atoms_per_frame), 3])
        # Use a dens number for indices .i.e 11, 17 -> 0, 1
        _species_dict = {symbol: i for i, symbol in enumerate(species_dict)}

        index = 0
        for i_frame, frame in enumerate(frame_generator):

            number_of_atoms = num_atoms_per_frame[i_frame]

            for i_atom, atom in enumerate(frame):
                self.representation_info[index, 0] = i_frame
                self.representation_info[index, 1] = i_atom
                self.representation_info[index, 2] = atom.number
                index += 1

            #TODO: fill with logic
            self.gradients_info = np.zeros([len(self.feature_gradients),5])

            results = self._transform_single_frame(frame, _species_dict)

            # Returned values are only the features
            if not self.compute_gradients:
                self.features[current_index:current_index+number_of_atoms] += results
            # Returned values are features + gradients
            else:
                self.features[current_index:current_index+number_of_atoms] += results[0]
                self.feature_gradients[gradient_index:gradient_index+number_of_atoms**2] += results[1]

            current_index += number_of_atoms
            gradient_index += number_of_atoms**2

    def _transform_single_frame(self, frame, species_dict):
        """
        Compute features for single frame and return to the transform()
        method which loops over all structures to obtain the complete
        vector for all environments.
        """
        # Initialization
        lmax = self.max_angular
        nmax = self.max_radial
        num_lm = (lmax+1)**2
        num_atoms = len(frame)
        num_chem_species = len(species_dict)
        iterator_species = np.zeros(num_atoms)
        for i, symbol in enumerate(frame.get_chemical_symbols()):
            iterator_species[i] = species_dict[symbol]

        # Initialize arrays in which to store all features
        frame_features = np.zeros((num_atoms, num_chem_species,
                                  self.max_radial, (self.max_angular+1)**2))
        if self.compute_gradients:
            num_gradients = np.sum(num_atoms**2)
            frame_gradients = np.zeros((num_gradients, 3, num_chem_species,
                                        self.max_radial, (self.max_angular+1)**2))

        # Extra phase dependent on angular channel for convenience
        angular_phases = np.zeros(lmax+1)
        for l in range(lmax+1):
            if l % 2 == 0:
                angular_phases[l] = (-1)**(l//2)
            else:
                angular_phases[l] = (-1)**((l+1)//2)

        ###
        # Step 1: Cell dependent quantities:
        #   In this step, we precompute the values of all factors that only
        #   depend on the cell. Most importantly, this includes the values
        #   of all functions that only depend on the k-vector. We use the
        #   same notation as in the original 2019 paper
        #   (more precisely, the section 2 in supplementary infonformation)
        ###
        # Get k-vectors (also called reciprocal space or Fourier vectors)
        kvecgen = KvectorGenerator(frame.get_cell(), np.pi / self.smearing)
        kvectors = kvecgen.get_kvectors()
        kvecnorms = kvecgen.get_kvector_norms()
        num_kvecs = kvecgen.get_kvector_number()

        # Fourier transform of density times Fourier transform of potential
        # This is the line where using Gaussian or 1/r^p for different p are
        # distinguished
        G_k = np.exp(-0.5 * (kvecnorms*self.smearing)**2)
        if self.potential_exponent == 1:
            G_k *= 4 * np.pi / kvecnorms**2

        # Spherical harmonics evaluated at the k-vectors
        # for angular projection
        Y_lm = evaluate_spherical_harmonics(kvectors, lmax)

        # Radial projection of spherical Bessel functions onto radial basis
        I_nl = self.radial_proj.radial_spline(kvecnorms)

        # Combine all these factors into single array
        # k_dep_factor is the combined k-dependent part
        k_dep_factor = np.zeros((num_kvecs, nmax, num_lm))
        for l in range(lmax+1):
            for n in range(nmax):
                f = np.atleast_2d(G_k * I_nl[:,n,l]).T * Y_lm[:, l**2:(l+1)**2]
                k_dep_factor[:, n, l**2:(l+1)**2] = f

        ###
        # Step 2: Structure factors:
        #   In this step, we precompute the trigonometric functions that
        #   will be used repeatedly in the remaining code
        ###
        # cosines[i, j] = cos(k_i * r_j), same for sines
        positions = frame.get_positions()
        args = kvectors @ positions.T
        cosines = np.cos(args)
        sines = np.sin(args)

        # real and imaginary parts of structure factor
        # technically, this step scales as O(VN^2) = O(N^3), where V is the
        # system volume and N the number of particles.
        # In practice, this step makes up for <5% of computational cost.
        # (--> around 15-25% savings in time due to numpy over manual loop)
        strucfac_real = np.zeros((num_kvecs, num_atoms, num_atoms))
        strucfac_imag = np.zeros((num_kvecs, num_atoms, num_atoms))
        for i in range(num_atoms):
            for j in range(num_atoms):
                strucfac_real[:, i, j] = cosines[:,i] * cosines[:,j] + sines[:,i] * sines[:,j]
                strucfac_imag[:, i, j] = cosines[:,i] * sines[:,j] - sines[:,i] * cosines[:,j]

        ###
        # Step 3: Main loop:
        #   Iterate over all atoms to evaluate the projection coefficients
        ###
        global_factor = 4 * np.pi / frame.get_volume()
        struc_factor = np.zeros((lmax+1)**2)
        struc_factor_grad = np.zeros((lmax+1)**2)

        # Loop over center atom
        for icenter in range(num_atoms):
            # Loop over all atoms in the structure (including central atom)
            for ineigh in range(num_atoms):
                i_chem = int(iterator_species[ineigh]) # index describing chemical species

                if self.potential_exponent == 0: # add constant term
                    I_nl_zero = self.radial_proj.radial_spline(0)
                    I_nl_zero /= np.sqrt(4 * np.pi)
                    frame_features[icenter, i_chem, :, 0] += I_nl_zero[:,0] * global_factor

                # Loop over all k-vectors
                for ik, kvector in enumerate(kvectors):
                    fourier_real = strucfac_real[ik,icenter,ineigh]
                    fourier_imag = strucfac_real[ik,icenter,ineigh]

                    # Phase factors depending on parity of l
                    for l in range(lmax+1):
                        if l % 2 == 0:
                            struc_factor[l**2:(l+1)**2] = angular_phases[l] * fourier_real
                        else:
                            struc_factor[l**2:(l+1)**2] = angular_phases[l] * fourier_imag

                    # Update features
                    # print('Current features =\n', np.round(frame_features[icenter, i_chem].T, 8))
                    # print('Additional term =\n', np.round(global_factor * struc_factor * k_dep_factor[ik], 8).T)
                    frame_features[icenter, i_chem] += 2 * global_factor * struc_factor * k_dep_factor[ik]
                    # print('New features =\n', np.round(frame_features[icenter, i_chem].T, 8))

                    # Update gradients
                    if self.compute_gradients:
                        # Phase factors depending on parity of l for gradients
                        for angular_l in range(lmax+1):
                            if angular_l % 2 == 0:
                                struc_factor_grad[l**2:(l+1)**2] = angular_phases[l] * fourier_imag
                            else:
                                struc_factor_grad[l**2:(l+1)**2] = angular_phases[l] * fourier_real

                        # Update x,y,z components
                        frame_gradients[ineigh + icenter * num_atoms, 0, i_chem] += global_factor * struc_factor_grad * k_dep_factor[ik] * kvector[0]
                        frame_gradients[ineigh + icenter * num_atoms, 1, i_chem] += global_factor * struc_factor_grad * k_dep_factor[ik] * kvector[1]
                        frame_gradients[ineigh + icenter * num_atoms, 2, i_chem] += global_factor * struc_factor_grad * k_dep_factor[ik] * kvector[2]

        if self.subtract_self:
            # Only for l = 0 and a = a_center
            for icenter in range(num_atoms):
                center_contrib = self.radial_projection.center_contributions
                frame_features[icenter, icenter, :, 0] -= center_contrib

        if self.compute_gradients:
            return frame_features, frame_gradients
        else:
            return frame_features


def run_example():
    from ase import Atoms
    from ase.io import read, write
    import time

    # frames = []
    # cell = np.eye(3) *12
    # distances = np.linspace(1.5, 3.5, 20)
    # for d in distances:
    #     # positions = [[0,0,0],[0,0,d],[0,d,0],[0,d,d],[d,d,d],[d,0,d],[d,0,0],[d,d,0]]
    #     # frame = Atoms('O8', positions=positions, cell=cell, pbc=True)

    #     positions2 = [[0,0,0],[0,0,d],[0,d,0],[0,d,d],[d,d,d]]
    #     frame = Atoms('BaTiO3', positions=positions2, cell=cell, pbc=True)
    #     frames.append(frame)

    # write('BaTiO3_toy_structures.xyz', frames)

    frames = read('BaTiO3_toy_structures.xyz', ':')

    # Define hyperparameters
    hypers = {
        'smearing':2.0,
        'max_angular':6,
        'cutoff_radius':3.5,
        'potential_exponent':1,
        'compute_gradients':True
        }

    species_dict = {'O':0, 'Ti':1, 'Ba':2}

    tstart = time.time()
    calculator = DensityProjectionCalculator(**hypers)
    calculator.transform(frames, species_dict)
    features = calculator.get_features()
    gradients = calculator.get_feature_gradients()

    features_ref = np.load('features_ref.npy')
    gradients_ref = np.load('gradients_ref.npy')

    err1 = np.linalg.norm(features-features_ref)
    err2 = np.linalg.norm(gradients-gradients_ref)
    print('Errors = ', err1, err2)
    print('Shapes = ', features.shape, gradients.shape)
    print(features[:5,0,0,-5:])
    tend = time.time()
    dt = tend - tstart
    print(f'Required time for {len(frames)} frames = {dt}s')


def run_example_gaussian():
    from ase import Atoms
    from ase.io import read, write
    import time

    frames = []
    cell = np.eye(3) *12
    distances = np.linspace(1.5, 2., 5)
    # for d in distances:
    #     positions2 = [[0,0,0],[0,0,d],[0,d,0],[0,d,d],[d,d,d]]
    #     frame = Atoms('O5', positions=positions2, cell=cell, pbc=True)
    #     frames.append(frame)
    # write('oxygen_toy_structures.xyz', frames)

    for d in distances:
        positions2 = [[1,1,1],[1,1,d+1]]
        frame = Atoms('O2', positions=positions2, cell=cell, pbc=True)
        frames.append(frame)


    # frames = read('oxygen_toy_structures.xyz', ':')

    # Define hyperparameters
    hypers = {
        'smearing':1.5,
        'max_angular':5,
        'max_radial':8,
        'cutoff_radius':5.,
        'potential_exponent':0,
        'compute_gradients':False
        }

    species_dict = {'O':0}

    tstart = time.time()
    calculator = Density_Projection_Calculator(**hypers)
    calculator.transform(frames, species_dict)
    features = calculator.get_features()
    # gradients = calculator.get_feature_gradients()

    # np.save('features_oxygen.npy', features)
    # np.save('gradients_oxygen.npy', gradients)

    # print('Shapes = ', features.shape)
    # print(np.round(features[-1,0,:].T,4))
    # tend = time.time()
    # dt = tend - tstart
    # print(f'Required time for {len(frames)} frames = {dt}s')


if __name__ == '__main__':
    # run_example()
    run_example_gaussian()
