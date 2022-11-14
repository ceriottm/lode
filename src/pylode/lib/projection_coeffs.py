# -*- coding: utf-8 -*-
"""

Python version of the LODE implementation.
Currently, only the exponent p=1 with the optimal radial basis r^l for each
angular channel l=0,1,2,...,lmax is supported.

"""

import logging

import numpy as np
from scipy.special import gamma
from scipy.integrate import quad

try:
    from tqdm import tqdm
except ImportError:
    tqdm = (lambda i, **kwargs: i)

from .kvec_generator import KvectorGenerator
from .radial_basis import RadialBasis
from .spherical_harmonics import evaluate_spherical_harmonics
from .atomic_density import AtomicDensity

logger = logging.getLogger(__name__)


class DensityProjectionCalculator:
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
        The radial basis. Currently implemented are
        'GTO_primitive', 'GTO', 'monomial'.
        For monomial: Only use one radial basis r^l for each angular
        channel l leading to a total of (lmax+1)^2 features.
    kcut : float
        Cutoff for the kspcae sum. If `None` it is set to `1.2 * π / smearing`
        which is a reasonable estimate for many systems.
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
                 cutoff_radius,
                 smearing,
                 radial_basis,
                 kcut=None,
                 compute_gradients=False,
                 potential_exponent=1,
                 subtract_center_contribution=False,
                 fast_implementation=True):
        # Store the input variables
        self.max_radial = max_radial
        self.max_angular = max_angular
        self.cutoff_radius = cutoff_radius
        self.radial_basis = radial_basis.lower()
        self.smearing = smearing
        self.potential_exponent = potential_exponent
        self.compute_gradients = compute_gradients
        self.subtract_center_contribution = subtract_center_contribution
        self.fast_implementation = fast_implementation
        if kcut is None:
            self.kcut = 1.2 * np.pi / smearing
        else:
            self.kcut = kcut

        # Make sure that the provided parameters are consistent
        if self.potential_exponent not in [0, 1, 2, 3, 4, 5, 6]:
            raise ValueError("Potential exponent has to be 0, 1, 2, ..., 6")

        if self.radial_basis not in ["monomial", "gto", "gto_primitive", "gto_analytical"]:
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
            cutoff_radius=self.cutoff_radius,
            smearing=self.smearing,
            radial_basis=self.radial_basis,
            subtract_self=self.subtract_center_contribution,
            potential_exponent=self.potential_exponent)

        self.radial_proj.compute(self.kcut)

        # Initialize atomic density class
        self.atomic_density = AtomicDensity(smearing, potential_exponent)

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
        # and assigned the array indices 0, 1, 2,...
        # Example: for H2O: H is mapped to 0 and O is mapped to 1.
        species = set()
        for frame in frames:
            for atom in frame:
               species.add(atom.number)
        species = sorted(species)
        self.species_dict = {}
        for frame in frames:
           # Get atomic species in dataset
           self.species_dict.update({atom.symbol: species.index(atom.number) for atom in frame})

        # Define variables determining size of feature vector coming from frames
        self.num_atoms_per_frame = np.array([len(frame) for frame in frames])
        num_atoms_total = np.sum(self.num_atoms_per_frame)
        num_chem_species = len(self.species_dict)

        # Initialize arrays in which to store all features
        self.features = np.zeros((num_atoms_total, num_chem_species,
                                  self.max_radial, (self.max_angular+1)**2))
        if self.compute_gradients:
            num_gradients = np.sum(self.num_atoms_per_frame**2)
            self.feature_gradients = np.zeros((num_gradients, 3, num_chem_species,
                                               self.max_radial, (self.max_angular+1)**2))

        # For each frame, compute the projection coefficients
        current_index = 0
        gradient_index = 0

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
                self.feature_gradients[gradient_index:gradient_index+number_of_atoms**2] += results[1]
            # Returned values are only the features
            else:
                features = results

            self.features[current_index:current_index+number_of_atoms] += features
            current_index += number_of_atoms
            gradient_index += number_of_atoms**2

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
        frame_features = np.zeros((num_atoms, num_chem_species,
                                self.max_radial, (self.max_angular+1)**2))
        if self.compute_gradients:
            num_gradients = np.sum(num_atoms**2)
            frame_gradients = np.zeros((num_gradients, 3, num_chem_species,
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
        kvecgen = KvectorGenerator(frame.get_cell(), self.kcut)
        kvecgen.compute()
        kvectors = kvecgen.kvectors
        kvecnorms = kvecgen.kvector_norms
        num_kvecs = kvecgen.kvector_number

        # Fourier transform of density times Fourier transform of potential
        # This is the line where using Gaussian or 1/r^p for different p are
        # distinguished
        G_k = self.atomic_density.get_fourier_transform(kvecnorms)

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
                strucfac_imag[:, i, j] = sines[:,i] * cosines[:,j] - cosines[:,i] * sines[:,j]
        
        strucfac_real *= 2
        strucfac_imag *= 2
 
        ###
        # Step 3: Main loop:
        #   Iterate over all atoms to evaluate the projection coefficients
        ###
        global_factor = 4 * np.pi / frame.get_volume()
        struc_factor = np.zeros(num_lm)
        struc_factor_grad = np.zeros(num_lm)
        struc_factor_all = np.zeros((num_lm, num_kvecs))
        struc_factor_grad_all = np.zeros((num_lm, num_kvecs))

        if self.compute_gradients and self.fast_implementation:
            kx = kvectors[:,0]
            ky = kvectors[:,1]
            kz = kvectors[:,2]

        # Loop over center atom
        for i_center in range(num_atoms):

            # index describing chemical species of center atom
            i_chem_center = iterator_species[i_center]

            # If desired (True by default), remove the contribution
            # of the center atom to the density.
            # By symmetry, this only affects the (l,m)=(0,0) components
            # of the projection coefficients and only the chemical
            # species channel that agrees with the center atom.
            if self.subtract_center_contribution:
                center_contrib = self.radial_proj.center_contributions.copy()
                frame_features[i_center, i_chem_center, :, 0] -= center_contrib

            # Loop over all atoms in the structure (including central atom)
            for i_neigh in range(num_atoms):

                # index describing chemical species of neighbor
                i_chem_neigh = iterator_species[i_neigh]

                # For Gaussian potentials, the Fourier transform
                # at k=0 is finite and contributes to the coefficients
                # (l,m)=(0,0). This is treated separately from the
                # remaining sum over k-points since it is only
                # used for specific densities and only affects (l,m)=(0,0).
                if self.potential_exponent == 0: # add constant term
                    I_nl_zero = self.radial_proj.radial_spline(0)
                    I_nl_zero /= np.sqrt(4 * np.pi)
                    I_nl_zero *= (2*np.pi*self.smearing**2)**1.5 / (np.pi*self.smearing**2)**(3/4)
                    frame_features[i_center, i_chem_neigh, :, 0] += I_nl_zero[:,0] * global_factor

                # For van der Waals interactions decaying as 1/r^6,
                # the potential decays fast enough that the Fourier
                # transform of the density at k=0 is well defined.
                # We add this contribution.
                elif self.potential_exponent in [4, 5, 6]:
                    peff = 3 - self.potential_exponent
                    prefac = np.pi / 2 * 2**peff / gamma(self.potential_exponent/2) # global prefactor appearing in Fourier transformed density
                    density_at_kzero = prefac * 2**((self.potential_exponent-1)/2) / (-peff) * self.smearing**(-peff) / self.smearing**(2*self.potential_exponent-6)
                    # density_at_kzero = self.atomic_density.get_fourier_transform_at_zero()
                    # TODO: redo this part to express the result in terms of the
                    # kspace density at zero obtained from the atomic_density_code
                    I_nl_zero = self.radial_proj.radial_spline(0)
                    I_nl_zero *= density_at_kzero
                    frame_features[i_center, i_chem_neigh, :, 0] += I_nl_zero[:,0] * global_factor

                # Slow implementation using manual loops:
                # this version is kept for better comparison with the C++ ver.
                if not self.fast_implementation:
                    # Loop over all k-vectors
                    for ik, kvector in enumerate(kvectors):
                        fourier_real = strucfac_real[ik, i_center, i_neigh]
                        fourier_imag = strucfac_imag[ik, i_center, i_neigh]
    
                        # Phase factors depending on parity of l
                        for l in range(lmax+1):
                            if l % 2 == 0:
                                struc_factor[l**2:(l+1)**2] = angular_phases[l] * fourier_real
                            else:
                                struc_factor[l**2:(l+1)**2] = angular_phases[l] * fourier_imag
    
                        frame_features[i_center, i_chem_neigh] += global_factor * struc_factor * k_dep_factor[ik]
    
                        # Update gradients
                        if self.compute_gradients:
                            # Phase factors depending on parity of l for gradients
                            for l in range(lmax+1):
                                if l % 2 == 0:
                                    struc_factor_grad[l**2:(l+1)**2] = angular_phases[l] * fourier_imag
                                else:
                                    struc_factor_grad[l**2:(l+1)**2] = -angular_phases[l] * fourier_real
    
                            # Update x,y,z components
                            i_grad = i_neigh + i_center * num_atoms
                            i_grad_center = i_center + i_center * num_atoms
                            frame_gradients[i_grad, 0, i_chem_neigh] += global_factor * struc_factor_grad * k_dep_factor[ik] * kvector[0]
                            frame_gradients[i_grad, 1, i_chem_neigh] += global_factor * struc_factor_grad * k_dep_factor[ik] * kvector[1]
                            frame_gradients[i_grad, 2, i_chem_neigh] += global_factor * struc_factor_grad * k_dep_factor[ik] * kvector[2]
                            frame_gradients[i_grad_center, 0, i_chem_neigh] -= global_factor * struc_factor_grad * k_dep_factor[ik] * kvector[0]
                            frame_gradients[i_grad_center, 1, i_chem_neigh] -= global_factor * struc_factor_grad * k_dep_factor[ik] * kvector[1]
                            frame_gradients[i_grad_center, 2, i_chem_neigh] -= global_factor * struc_factor_grad * k_dep_factor[ik] * kvector[2]
                    
                else: # use fast implementation
                    fourier_real = strucfac_real[:, i_center, i_neigh]
                    fourier_imag = strucfac_imag[:, i_center, i_neigh]

                    # Phase factors depending on parity of l
                    for l in range(lmax+1):
                        if l % 2 == 0:
                            struc_factor_all[l**2:(l+1)**2] = angular_phases[l] * fourier_real
                        else:
                            struc_factor_all[l**2:(l+1)**2] = angular_phases[l] * fourier_imag

                    contr = np.sum(k_dep_factor_reordered * struc_factor_all, axis=2)
                    frame_features[i_center, i_chem_neigh] += global_factor * contr
                    
                    # Update gradients
                    if self.compute_gradients and i_center != i_neigh:
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
                        i_grad = i_neigh + i_center * num_atoms
                        i_grad_center = i_center + i_center * num_atoms
                        frame_gradients[i_grad, 0, i_chem_neigh] += global_factor * gradx
                        frame_gradients[i_grad, 1, i_chem_neigh] += global_factor * grady 
                        frame_gradients[i_grad, 2, i_chem_neigh] += global_factor * gradz
                        frame_gradients[i_grad_center, 0, i_chem_neigh] -= global_factor * gradx
                        frame_gradients[i_grad_center, 1, i_chem_neigh] -= global_factor * grady 
                        frame_gradients[i_grad_center, 2, i_chem_neigh] -= global_factor * gradz

        if self.compute_gradients:
            return frame_features, frame_gradients
        else:
            return frame_features

    @property
    def representation_info(self):
        """Metadata about features. Same as in librascal.

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
