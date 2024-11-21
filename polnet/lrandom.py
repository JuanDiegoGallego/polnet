"""
Classes with the implementation of the random models.
A networks is a combination of a polymer in a volume
"""

__author__ = 'Antonio Martinez-Sanchez'

import math
import random
import numpy as np
from abc import ABC, abstractmethod

from vtkmodules.generate_pyi import has_self

MAX_TRIES_ELLIP = 1e6
MAX_TRIES_EXP = 1e6

#############################################
# Functions
#############################################


def gen_bounded_exp(mean, lb, ub):
    """
    Generates a random number following a 'bounded exponential distribution'
    Get random exponential numbers until falls into bounded range

    :param mean: mean for the exponential distribution (1/lambda)
    :param lb: lower bound
    :param hb: higher bound
    :return: a real number within the range, raises RuntimeError if no number found within range after MAX_TRIES_EXP
    """
    hold = np.random.exponential(scale=mean)
    count = 1
    while ((hold < lb) or (hold > ub)) and (count < MAX_TRIES_EXP):
        hold = np.random.exponential(scale=mean)
        count += 1
    if count >= MAX_TRIES_EXP:
        raise RuntimeError
    else:
        return hold

#############################################
# Classes
#############################################


class PGenHelixFiber(ABC):
    """
    Class to model random distribution for helix fiber parameters
    """

    def gen_length(self, min_l, max_l):
        """
        Generate a length with a range following a uniform distribution

        :param min_l: minimum length
        :param max_l: maximum length
        :return:
        """
        assert (min_l >= 0) and (min_l <= max_l)
        return (max_l - min_l) * random.random() + min_l

    def gen_persistence_length(self, min_p):
        """
        Generate a persistence length according an exponential distribution with lambda=1 and minimum value

        :param min_p: minimum persistence value
        :return:
        """
        return min_p + np.random.exponential()

    def gen_zf_length(self, min_zf=0, max_zf=1):
        """
        Generates a z-axis factor within a range

        :param min_zf: minimum value (default 0)
        :param max_zf: maximum value (default 1)
        """
        assert (min_zf >= 0) and (max_zf <= 1)
        return (max_zf - min_zf) * random.random() + min_zf


class PGenHelixFiberB(PGenHelixFiber):
    """
    Class to model random distribution for helix fiber parameters with branches
    """

    def gen_branch(self, b_prob=0.5):
        """
        Generates a boolean that is True (branching) with some input probability

        :param b_prob: branching probability [0, 1) (default 0.5)
        :return: a boolean
        """
        if random.random() <= b_prob:
            return True
        else:
            return False


class SurfGen(ABC):
    """
    Abstract class for modeling 3D parametric surface random parameter generators
    """

    def gen_den_cf(self, low, high):
        """
        Generates a uniform random real number between two input values

        :return:
        """
        return random.uniform(low, high)

    @abstractmethod
    def gen_parameters(self):
        raise NotImplemented


class EllipGen(SurfGen):
    """
    Class for model the paramaters for modelling an Ellipsoid
    """

    def __init__(self, radius_rg, max_ecc=1, max_tries=int(1e6)):
        """
        Constructor

        :param radius_rg: ranges for semi-axis parameters
        :param max_ecc: maximum eccentricity for both planes (default 1), in range [0, 1]
        :param max_tries: raises an RuntimeError exception if there is no proper settings are found before trying
                 'max_tries' times
        """
        assert hasattr(radius_rg, '__len__') and (len(radius_rg) == 2) and (radius_rg[0] <= radius_rg[1])
        assert (max_ecc >= 0) and (max_ecc <= 1)
        assert isinstance(max_tries, int) and (max_tries > 0)
        self.__radius_rg, self.__max_ecc, self.__max_tries = radius_rg, max_ecc, max_tries

    def gen_parameters(self):
        """
        Generates randomly the three semi-axes parameters following a uniform distribution

        :return: an array with the three semi-axes sorted in descending order
        """
        for i in range(self.__max_tries):
            axes = np.sort(np.asarray((random.uniform(self.__radius_rg[0], self.__radius_rg[1]),
                                       random.uniform(self.__radius_rg[0], self.__radius_rg[1]),
                                       random.uniform(self.__radius_rg[0], self.__radius_rg[1]))))[::-1]
            ecc1, ecc2 = math.sqrt(1 - (axes[2] / axes[0])**2), math.sqrt(1 - (axes[2] / axes[1])**2)
            if (ecc1 <= self.__max_ecc) and (ecc2 <= self.__max_ecc):
                return axes
        raise RuntimeError

    def gen_parameters_exp(self):
        """
        Generates randomly the three semi-axes parameters following a bounded exponential distribution

        :return: an array with the three semi-axes sorted in descending order
        """
        for i in range(self.__max_tries):
            axes = np.sort(np.asarray((gen_bounded_exp(8.*self.__radius_rg[0], self.__radius_rg[0], self.__radius_rg[1]),
                                       gen_bounded_exp(8.*self.__radius_rg[0], self.__radius_rg[0], self.__radius_rg[1]),
                                       gen_bounded_exp(8.*self.__radius_rg[0], self.__radius_rg[0], self.__radius_rg[1]))))[::-1]
            ecc1, ecc2 = math.sqrt(1 - (axes[2] / axes[0])**2), math.sqrt(1 - (axes[2] / axes[1])**2)
            if (ecc1 <= self.__max_ecc) and (ecc2 <= self.__max_ecc):
                return axes
        raise RuntimeError


class SphGen(SurfGen):
    """
    Class for model the parameters for modelling an Sphere
    """

    def __init__(self, radius_rg):
        """
        Constructor

        :param radius_rg: ranges for radius
        """
        assert hasattr(radius_rg, '__len__') and (len(radius_rg) == 2) and (radius_rg[0] <= radius_rg[1])
        self.__radius_rg = radius_rg

    def gen_parameters(self):
        """
        Generates randomly sphere radii

        :return: a float with the radius value
        """
        return random.uniform(self.__radius_rg[0], self.__radius_rg[1])


class TorGen(SurfGen):
    """
    Class for model the paramaters for modelling a Torus
    """

    def __init__(self, radius_rg):
        """
        Constructor

        :param radius_rg: ranges for radii parameters
        """
        assert hasattr(radius_rg, '__len__') and (len(radius_rg) == 2) and (radius_rg[0] <= radius_rg[1])
        self.__radius_rg = radius_rg

    def gen_parameters(self):
        """
        Generates randomly the two Torus radii

        :return: a array with the two radii ('a' torus radius, and 'b' tube radius, where a > b)
        """
        return np.sort(np.asarray((random.uniform(self.__radius_rg[0], self.__radius_rg[1]),
                                   random.uniform(self.__radius_rg[0], self.__radius_rg[1]))))[::-1]


class SGen(ABC):
    """
    Abstract class for a (random) sequence
    """

    @abstractmethod
    def gen_next_mmer_id(self, n_mmers, prev_id):
        """
        Generated the id for the next monomer

        :param n_mmers: number of diferent monomers
        :param prev_id: previous monomer identifier
        :return: an integer indication the next monomer
        """
        raise NotImplemented


class SGenFixed(SGen):
    """
    A fixed sequence sequence is generated
    """

    def gen_next_mmer_id(self, n_surf, prev_id):
        """
            Generated the id for the next monomer

            :param n_mmers: number of diferent monomers
            :param prev_id: previous monomer identifier
            :return: an integer indication the next monomer
        """
        curr_id = prev_id + 1
        if curr_id >= n_surf: return 0
        else: return curr_id


class SGenUniform(SGen):
    """
    Uniformly random distribution
    """

    def gen_next_mmer_id(self, n_surf, prev_id=None):
        """
            Generated the id for the next monomer

            :param n_mmers: number of diferent monomers
            :param prev_id: previous monomer identifier (default None, not needed because this model is memoryless)
            :return: an integer indication the next monomer
        """
        return random.randint(0, n_surf-1)


class SGenProp(SGen):
    """
    Generates numbers according to some proportion
    """

    def __init__(self, prop_l):
        """
        :param prop_l: proportions list
        """
        assert hasattr(prop_l, '__len__') and sum(prop_l) == 1
        self.__n_surf = len(prop_l)
        self.__props = prop_l
        self.__mmer_ids = range(len(self.__props))

    def gen_next_mmer_id(self, n_surf=None, prev_id=None):
        """
            Generated the id for the next monomer

            :param n_mmers: number of different monomers, (default None, not needed because this information is
            introduced in this model during construction)
            :param prev_id: previous monomer identifier (default None, not needed because this model is memoryless)
            :return: an integer indication the next monomer
        """
        return random.choices(self.__mmer_ids, weights=self.__props, k=1)[0]


class OccGen():
    """
    Class for model the random (uniform) generation of occupancies
    """

    def __init__(self, occ_rg):
        """
        Constructor

        :param occ_rg: range for occupancy
        """
        assert hasattr(occ_rg, '__len__') and (len(occ_rg) == 2) and (occ_rg[0] <= occ_rg[1])
        assert (occ_rg[0] >= 0) and (occ_rg[1] <= 100)
        self.__occ_rg = occ_rg

    def gen_occupancy(self):
        """
        Generates randomly an occupancy value

        :return: a float with the generated occupancy
        """
        return random.uniform(self.__occ_rg[0], self.__occ_rg[1])

# JUAN DIEGO
# TODO Pass more parameters for generation
class CvtGen(SurfGen):
    """
    Class for model the parameters for modeling Curvatubes # Curvatubes
    a20 always equals 1
    eps always equals 0.02
    """

    def __init__(self,mass_rg=(-0.7,0.3),a11_a02_rg=(-10,10),b10_b01_rg=(-100,100),c_rg=(-1000,1000)):
        """
        Constructor

        :param mass_rg: range for mass
        :param a11_a02_rg: range for a11 and a02
        :param b10_b01_rg: range for b10 and b01
        :param c_rg: range
        """
        assert hasattr(mass_rg, '__len__') and (len(mass_rg) == 2) and (mass_rg[0] <= mass_rg[1])
        assert hasattr(a11_a02_rg, '__len__') and (len(a11_a02_rg) == 2) and (a11_a02_rg[0] <= a11_a02_rg[1])
        assert hasattr(b10_b01_rg, '__len__') and (len(b10_b01_rg) == 2) and (b10_b01_rg[0] <= b10_b01_rg[1])
        assert hasattr(c_rg, '__len__') and (len(c_rg) == 2) and (c_rg[0] <= c_rg[1])
        self.__mass_rg = mass_rg
        self.__a11_a02_rg = a11_a02_rg
        self.__b10_b01_rg = b10_b01_rg
        self.__c_rg = c_rg

    def gen_parameters(self):
        """
        Generates randomly the parameters needed for the generation of a curvatube following bounded uniform distributions

        :return: the eight parameters needed
        """
        a20 = 1
        eps = 0.02
        m = random.uniform(self.__mass_rg[0], self.__mass_rg[1])
        a11 = random.uniform(self.__a11_a02_rg[0], self.__a11_a02_rg[1])
        a02 = random.uniform(self.__a11_a02_rg[0], self.__a11_a02_rg[1])
        b10 = random.uniform(self.__b10_b01_rg[0], self.__b10_b01_rg[1])
        b01 = random.uniform(self.__b10_b01_rg[0], self.__b10_b01_rg[1])
        c = random.uniform(self.__c_rg[0], self.__c_rg[1])

        return eps, a20, a11, a02, b10, b01, c, m

