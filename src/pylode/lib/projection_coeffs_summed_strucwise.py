# -*- coding: utf-8 -*-
"""

Python version of the LODE implementation.
Currently, only the exponent p=1 with the optimal radial basis r^l for each
angular channel l=0,1,2,...,lmax is supported.

"""

import logging

import numpy as np
from scipy.integrate import quad

try:
    from tqdm import tqdm
except ImportError:
    tqdm = (lambda i, **kwargs: i)

from .kvec_generator import KvectorGenerator
from .radial_basis import RadialBasis
from .spherical_harmonics import evaluate_spherical_harmonics

logger = logging.getLogger(__name__)

def gammainc_upper_numerical(n, zz):
    """
    Implement upper incomplete Gamma function
    """
    yy = np.zeros_like(zz)
    integrand = lambda x: x**(n-1) * np.exp(-x)
    for iz, z in enumerate(zz):
        yy[iz] = quad(integrand, z, np.inf)[0]
    return yy

class DensityProjectionCalculatorSummed:
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
    radial_basis_radius : float
        Environment cutoff (Å)
    smearing : float
        Smearing of the Gaussain (Å). Note that computational cost scales
        cubically with 1/smearing.
    radial_basis : str
        The radial basis. Currently implemented are
        'GTO_primitive', 'GTO', 'monomial'.
        For monomial: Only use one radial basis r^l for each angular
        channel l leading to a total of (lmax+1)^2 features.
    compute_gradients : bool
        Compute gradients
    potential_exponent : int
        potential exponent: p=0 uses Gaussian densities,
        p=1 is LODE using 1/r (Coulomb) densities"
    subtract_center_contribution : bool
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
        Metadata to interact with equsitore.
    """
    def __init__(self,
                 max_radial,
                 max_angular,
                 radial_basis_radius,
                 smearing,
                 radial_basis,
                 compute_gradients=False,
                 potential_exponent=1,
                 subtract_center_contribution=False,
                 fast_implementation=True):
        # Store the input variables
        self.max_radial = max_radial
        self.max_angular = max_angular
        self.radial_basis_radius = radial_basis_radius
        self.radial_basis = radial_basis.lower()
        self.smearing = smearing
        self.potential_exponent = potential_exponent
        self.compute_gradients = compute_gradients
        self.subtract_center_contribution = subtract_center_contribution
        self.fast_implementation = fast_implementation

        # Make sure that the provided parameters are consistent
        if self.potential_exponent not in [0, 1, 2, 3, 4, 5, 6]:
            raise ValueError("Potential exponent has to be 0, 1, 2, ..., 6")

        if self.radial_basis not in ["monomial", "gto", "gto_primitive"]:
            raise ValueError(f"{self.radial_basis} is not an implemented basis"
                              ". Try 'monomial', 'GTO' or GTO_primitive.")

        if self.radial_basis == "monomial" and self.max_radial != 1:
            raise ValueError("For monomial basis only `max_radial=1` "
                             "is allowed.")

        # Auxilary quantity: the actual number of features is this number
        # times the number of chemical species
        self.num_features_bare = self.max_radial * (self.max_angular + 1)**2

        # Initialize radial basis class to precompute the quantities
        # only related to the choice of radial basis, namely the
        # projections of the spherical Bessel functions j_l(kr) onto the
        # radial basis and (if desired) the center contributions
        self.radial_proj = RadialBasis(max_radial = self.max_radial,
            max_angular=self.max_angular,
            radial_basis_radius=self.radial_basis_radius,
            smearing=self.smearing,
            radial_basis=self.radial_basis,
            subtract_self=self.subtract_center_contribution,
            potential_exponent=self.potential_exponent)
        self.radial_proj.compute(2*np.pi/self.smearing)

    def transform(self, frames, show_progress=False):
        """
        Computes the features and (if compute_gradients == True) gradients
        for all the provided frames. The features and gradients are stored in
        features and feature_gradients attribute.

        Parameters
        ----------
        frames : ase.Atoms
            List containing all ase.Atoms structures
        show_progress : bool
            Show progress bar for frame analysis

        Returns
        -------
        None, but stores the projection coefficients and (if desired)
        gradients as arrays as `features` and `features_gradients`.
        """
        self.frames = frames

        # Check that the provided cells are large enough:
        # Roughly speaking, all cell dimensions L need to be at least
        # twice the used smearing: L > 2 * smearing
        too_small_frames_list = []
        length_min = 1e15
        for iframe, frame in enumerate(frames):
            cell = frame.get_cell()
            basis_vector_lengths = np.linalg.norm(cell, axis=1)
            length_min_cell = max(basis_vector_lengths)
            if length_min > length_min_cell:
                length_min = length_min_cell
            if 2*self.smearing >= length_min_cell:
                too_small_frames_list.append(iframe)

        if self.smearing >= length_min/2:

            raise ValueError(f"Given `smearing` ({self.smearing} Å) is too large for "
                            f"structures {too_small_frames_list}. Smearing must be"
                            f"smaller than half of the shortest"
                            f"box length ({length_min} Å)! "
                            f"Use a smearing > {length_min/2}")

        # Generate a dictionary to map atomic species to array indices
        # In general, the species are sorted according to atomic number
        # and assigned the array indices 0,1,2,...
        # Example: for H2O: H is mapped to 0 and O is mapped to 1.
        species = set()
        for frame in frames:
            for atom in frame:
               species.add(atom.number)
        species = sorted(species)
        self.species_dict = {}
        for frame in frames:
            #Get atomic species in dataset
           self.species_dict.update({atom.symbol: species.index(atom.number) for atom in frame})

        # Define variables determining size of feature vector coming from frames
        self.num_atoms_per_frame = np.array([len(frame) for frame in frames])
        num_atoms_total = np.sum(self.num_atoms_per_frame)
        num_chem_species = len(self.species_dict)

        # Initialize arrays in which to store all features
        self.features = np.zeros((len(frames), num_chem_species, num_chem_species,
                                  self.max_radial, (self.max_angular+1)**2))
        if self.compute_gradients:
            self.feature_gradients = np.zeros((num_atoms_total, 3, num_chem_species, num_chem_species,
                                               self.max_radial, (self.max_angular+1)**2))

        # For each frame, compute the projection coefficients
        atom_index = 0

        if show_progress:
            frame_generator = tqdm(self.frames)
        else:
            frame_generator = self.frames

        for i_frame, frame in enumerate(frame_generator):

            number_of_atoms = self.num_atoms_per_frame[i_frame]
            results = self._transform_single_frame(frame)

            # Returned values are features + gradients
            if self.compute_gradients:
                features = results[0]
                self.feature_gradients[atom_index:atom_index+number_of_atoms] += results[1]
            # Returned values are only the features
            else:
                features = results

            self.features[i_frame] += features
            atom_index += number_of_atoms


    def _transform_single_frame(self, frame):
        """
        Compute features for single frame and return to the transform()
        method which loops over all structures to obtain the complete
        vector for all environments.
        """
        ###
        # Initialization
        ###
        # Define useful shortcuts
        lmax = self.max_angular
        nmax = self.max_radial
        num_lm = (lmax+1)**2
        num_atoms = len(frame)
        num_chem_species = len(self.species_dict)
        iterator_species = np.zeros(num_atoms, dtype=int)
        for i, symbol in enumerate(frame.get_chemical_symbols()):
            iterator_species[i] = self.species_dict[symbol]

        # Initialize arrays in which to store all features
        frame_features = np.zeros((num_chem_species, num_chem_species,
                                self.max_radial, (self.max_angular+1)**2))
        if self.compute_gradients:
            frame_gradients = np.zeros((num_atoms, 3, num_chem_species, num_chem_species,
                                        self.max_radial, (self.max_angular+1)**2))

        # Debug log
        logger.debug(f"num_atoms = {num_atoms}")
        logger.debug(f"shape frame_features = {frame_features.shape}")

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
        kvecgen = KvectorGenerator(frame.get_cell(), 1.2 * np.pi / self.smearing)
        kvecgen.compute()
        kvectors = kvecgen.kvectors
        kvecnorms = kvecgen.kvector_norms
        num_kvecs = kvecgen.kvector_number

        # Fourier transform of density times Fourier transform of potential
        # This is the line where using Gaussian or 1/r^p for different p are
        # distinguished
        if self.potential_exponent == 0:
            prefac = (4 * np.pi * self.smearing**2)**(3/4)
            G_k = prefac * np.exp(-0.5 * (kvecnorms*self.smearing)**2)
        elif self.potential_exponent == 1:
            G_k = 4 * np.pi / kvecnorms**2 * np.exp(-0.5 * (kvecnorms*self.smearing)**2)
        else:
            prefac = 4 * np.pi
            smeareff = self.potential_exponent * self.smearing
            peff = 3 - self.potential_exponent 
            G_k = prefac * gammainc_upper_numerical(peff/2, 0.5 * (kvecnorms*smeareff)**2)
            G_k /= kvecnorms**peff

        # Spherical harmonics evaluated at the k-vectors
        # for angular projection
        Y_lm = evaluate_spherical_harmonics(kvectors, lmax)

        # Radial projection of spherical Bessel functions onto radial basis
        I_nl = self.radial_proj.radial_spline(kvecnorms)

        # Combine all these factors into single array
        # k_dep_factor is the combined k-dependent part
        k_dep_factor = np.zeros((num_kvecs, nmax, num_lm))
        k_dep_factor_reordered = np.zeros((nmax, num_lm, num_kvecs))
        for l in range(lmax+1):
            for n in range(nmax):
                f = np.atleast_2d(G_k * I_nl[:,n,l]).T * Y_lm[:, l**2:(l+1)**2]
                k_dep_factor[:, n, l**2:(l+1)**2] = f
                k_dep_factor_reordered[n, l**2:(l+1)**2, :] = f.T

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

        # Center species-wise summed up version of structure factors
        strucfac_real_summed = np.zeros((num_kvecs, num_chem_species, num_atoms))
        strucfac_imag_summed = np.zeros((num_kvecs, num_chem_species, num_atoms))
        for a_center in range(num_chem_species):
            indices_species = (iterator_species == a_center)
            cos_summed = np.sum(cosines[:,indices_species], axis=1)
            sin_summed = np.sum(sines[:,indices_species], axis=1)
            for j in range(num_atoms):
                strucfac_real_summed[:, a_center, j] = cos_summed * cosines[:,j] + sin_summed * sines[:,j]
                strucfac_imag_summed[:, a_center, j] = sin_summed * cosines[:,j] - cos_summed * sines[:,j]
    
        ###
        # Step 3: Main loop:
        #   Iterate over all atoms to evaluate the projection coefficients
        ###
        global_factor = 4 * np.pi / frame.get_volume()
        struc_factor_all = np.zeros((num_lm, num_kvecs))
        struc_factor_grad_all = np.zeros((num_lm, num_kvecs))

        if self.compute_gradients and self.fast_implementation:
            kx = kvectors[:,0]
            ky = kvectors[:,1]
            kz = kvectors[:,2]

        # Loop over center atom
        for a_center in range(num_chem_species):
            indices_species = (iterator_species == a_center)
            num_this_species = np.sum(indices_species)
            
            if self.subtract_center_contribution:
                center_contrib = self.radial_proj.center_contributions.copy()
                center_contrib *= num_this_species
                frame_features[a_center, a_center, :, 0] -= center_contrib
            
            for i_neigh in range(num_atoms):
                # index describing chemical species of neighbor
                i_chem_neigh = iterator_species[i_neigh]

                # For Gaussian potentials, the Fourier transform
                # at k=0 is finite and contributes to the coefficients
                # (l,m)=(0,0). This is treated separately from the
                # remaining sum over k-points since it is only
                # used for specific densities and only affects (l,m)=(0,0).
                if self.potential_exponent == 0: # add constant term
                    I_nl_zero = self.radial_proj.radial_spline(0) * num_this_species
                    I_nl_zero /= np.sqrt(4 * np.pi)
                    I_nl_zero *= (2*np.pi*self.smearing**2)**1.5 / (np.pi*self.smearing**2)**(3/4)
                    frame_features[a_center, i_chem_neigh, :, 0] += I_nl_zero[:,0] * global_factor

                    
                # use fast implementation
                fourier_real = strucfac_real_summed[:, a_center, i_neigh]
                fourier_imag = strucfac_imag_summed[:, a_center, i_neigh]

                # Phase factors depending on parity of l
                for l in range(lmax+1):
                    if l % 2 == 0:
                        struc_factor_all[l**2:(l+1)**2] = angular_phases[l] * fourier_real
                    else:
                        struc_factor_all[l**2:(l+1)**2] = angular_phases[l] * fourier_imag

                # Add up contributions
                # The factor of 2 comes from pairs of the form +k and -k
                # which are grouped together
                contr = np.sum(k_dep_factor_reordered * struc_factor_all, axis=2)
                frame_features[a_center, i_chem_neigh] += 2 * global_factor * contr
                
                # Update gradients
                if self.compute_gradients:
                    # Phase factors depending on parity of l for gradients
                    for l in range(lmax+1):
                        if l % 2 == 0:
                            struc_factor_grad_all[l**2:(l+1)**2] = angular_phases[l] * fourier_imag
                        else:
                            struc_factor_grad_all[l**2:(l+1)**2] = -angular_phases[l] * fourier_real

                    # Update x,y,z components
                    gradx = np.sum(k_dep_factor_reordered * struc_factor_grad_all * kx, axis=2)
                    grady = np.sum(k_dep_factor_reordered * struc_factor_grad_all * ky, axis=2)
                    gradz = np.sum(k_dep_factor_reordered * struc_factor_grad_all * kz, axis=2)
                    i_grad = i_neigh
                    #i_grad_center = i_center + i_center * num_atoms
                    #frame_gradients[i_grad, 0, a_center, i_chem_neigh] += 2 * global_factor * gradx
                    #frame_gradients[i_grad, 1, a_center, i_chem_neigh] += 2 * global_factor * grady 
                    #frame_gradients[i_grad, 2, a_center, i_chem_neigh] += 2 * global_factor * gradz
                    #frame_gradients[i_grad_center, 0, i_chem_neigh] -= 2 * global_factor * gradx
                    #frame_gradients[i_grad_center, 1, i_chem_neigh] -= 2 * global_factor * grady 
                    #frame_gradients[i_grad_center, 2, i_chem_neigh] -= 2 * global_factor * gradz

        if self.compute_gradients:
            return frame_features, frame_gradients
        else:
            return frame_features

    @property
    def representation_info(self):
        """Metadata about features. Same as in librascal.
        TODO: Adjust this to new shapes

        Returns
        -------
        np.array of size (n_atoms, 3)
            The array has as many rows as the number of representations
            and they correspond to the index of the structure, the central atom
            and its atomic species.
        """

        representation_info = []
        for i_frame, frame in enumerate(self.frames):
            for i_center, center_atom in enumerate(frame):
                representation_info.append(
                    (i_frame, i_center, center_atom.number))

        return np.array(representation_info, dtype=np.int32)

    @property
    def gradients_info(self):
        """Metadata about gradients. Same as in librascal.
        TODO: Adjust this to new shapes

        Returns
        -------
        np.array of size (n_frames * (n_neighbors + n_atoms), 5)

            The array has as many rows as as the number gradients and each
            columns correspond to the index of the atomic structure,
            central atom, the neighbor atom and their atomic species.
        """

        gradients_info = []
        for i_frame, frame in enumerate(self.frames):
            num_atoms = len(frame)

            for i_center in range(num_atoms):
                center_atom = frame[i_center]

                for i_neigh in range(num_atoms):
                    neigh_atom = frame[i_neigh]

                    gradients_info_frame = (i_frame, i_center, i_neigh,
                                            center_atom.number,
                                            neigh_atom.number)

                    gradients_info.append(gradients_info_frame)

        return np.array(gradients_info)
