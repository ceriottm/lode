# -*- coding: utf-8 -*-
"""
Real space implementation of the projection coefficients.

As opposed to the reciprocal space implementation, this version only
takes into account the contributions of neighbors up to a provided
cutoff distance.
The main intended usage for now is to remove the contributions from all
atoms within the cutoff from the full reciprocal space coefficients to
obtain a true multiscale model.

"""

import logging

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = (lambda i, **kwargs: i)

from .radial_basis import RadialBasis
from .spherical_harmonics import evaluate_spherical_harmonics
from .neighbor_list import NeighborList

logger = logging.getLogger(__name__)


class DensityProjectionCalculatorRealspace:
    """
    Compute the spherical expansion coefficients in realspace.

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

        logger.info('Start real space implementation')
        # Make sure that the provided parameters are consistent
        if self.potential_exponent not in [0, 1]:
            raise ValueError("Potential exponent has to be one of 0 or 1!")

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
        self.radial_proj = RadialBasis(self.max_radial,
                                       self.max_angular,
                                       self.cutoff_radius,
                                       self.smearing,
                                       self.radial_basis,
                                       potential_exponent,
                                       self.subtract_center_contribution)
        self.radial_proj.compute(1.0)
        logger.info("Precalculate splines for radial integral. "
                    "This might take a while...")
        self.radial_proj.compute_realspace_spline_from_analytical()

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
            logger.info(f'Frame number = {i_frame}')
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
	
        ###
        #   Iterate over all atoms to evaluate the projection coefficients
        ###
        global_factor = 2 * np.pi
        struc_factor = np.zeros(num_lm)
        struc_factor_grad = np.zeros(num_lm)
        neighbor_list = NeighborList(frame, self.species_dict, self.cutoff_radius)

        # Loop over center atom
        for i_center in range(num_atoms):

            # Loop over all possible neighbor species
            neighbors_i = neighbor_list.neighbor_list[i_center]

            # neighbors_i is a list, where the 0-th entry contains
            # the neighbor information about all neighbors of "species 0",
            # the next entry about all neighbors of "species 1", etc.,
            # where species 0,1,2,... are defined in the species_dict
            # Thus, looping over the entries of neighbors_i separates
            # the different chemical species automatically.
            for i_chem_neigh, neighbors_ia in enumerate(neighbors_i):
                # Check whether neighbors of this species are present
                if neighbors_ia.entries['number_of_neighbors'] == 0:
                    continue

                # Radial contributions
                distances = neighbors_ia.entries['pair_distances']
                I_nl = self.radial_proj.radial_spline_realspace(distances) # shape N x nmax x (lmax+1)

                # Angular contributions
                vectors = neighbors_ia.entries['pair_vectors']
                sph_harm = evaluate_spherical_harmonics(vectors, lmax).T # shape N x (lmax+1)^2

                for n in range(nmax):
                    I_nl_n_fixed = I_nl[:, n, :].T
                    for l in range(lmax+1):
                        summand = I_nl_n_fixed[l] * sph_harm[l**2:(l+1)**2]
                        coeff = np.sum(summand, axis=1)
                        frame_features[i_center, i_chem_neigh, n, l**2:(l+1)**2] += coeff
                        

            # If required: set gradients to correct values 
            # TODO: implement this
            if self.compute_gradients:
                for i_neigh in range(num_atoms):
                    # Update x,y,z components
                    frame_gradients[i_neigh + i_center * num_atoms, 0, i_chem_neigh] += global_factor
                    frame_gradients[i_neigh + i_center * num_atoms, 1, i_chem_neigh] += global_factor
                    frame_gradients[i_neigh + i_center * num_atoms, 2, i_chem_neigh] += global_factor
        
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
