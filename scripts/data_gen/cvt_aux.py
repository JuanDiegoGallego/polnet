import numpy as np
import math
import scipy as sp
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from polnet.utils import lin_map, density_norm


# Auxiliary functions
def gen_blyr_basic(ct_den, lyr_dist, lyr_thick, vx_sz, sigma):
    """
        Create a MRC bilayer tomogram from a density map using the distance transform

        :param ct_den: the input density map
        :param lyr_dist: the distance of the layers to the center of the membrane
        :param lyr_thick: the thickness of the layers
        :return: the ndarray representing the bilayers after applying a gaussian filter
        """
    # Generating distance-to-layer-map
    ct_dist_in = np.float32(distance_transform_edt(ct_den))
    ct_dist_out = np.float32(distance_transform_edt(-ct_den + 1))
    ct_dist_in *= vx_sz
    ct_dist_out *= vx_sz
    ct_dist = ct_dist_in - ct_dist_out
    del ct_dist_in
    del ct_dist_out

    # Taking the max value of the array
    ct_max_v = ct_dist.max()

    thr_o = -lyr_dist
    ct_lyr_o = np.copy(ct_dist)
    ct_lyr_o[(ct_lyr_o >= thr_o - lyr_thick) & (ct_lyr_o <= thr_o + lyr_thick)] = ct_max_v + 1
    ct_lyr_o[ct_lyr_o < ct_max_v] = 0
    ct_lyr_o[ct_lyr_o > 0] = 1

    thr_i = lyr_dist
    ct_lyr_i = np.copy(ct_dist)
    del ct_dist
    ct_lyr_i[(ct_lyr_i >= thr_i - lyr_thick) & (ct_lyr_i <= thr_i + lyr_thick)] = ct_max_v + 1
    ct_lyr_i[ct_lyr_i < ct_max_v] = 0
    ct_lyr_i[ct_lyr_i > 0] = 1

    ct_blyr = ct_lyr_o + ct_lyr_i
    #return ct_blyr
    ct_blyr = lin_map(density_norm(gaussian_filter(ct_blyr, sigma), inv=True), ub=0, lb=1)
    return ct_blyr


def gen_blyr_exp(ct_den, lyr_dist, lyr_thick, vx_sz, sigma, fact):
    """
        Create a MRC bilayer tomogram from a density map using the distance transform after scaling by fact the tomogram

        :param ct_den: the input density map
        :param lyr_dist: the distance of the layers to the center of the membrane
        :param lyr_thick: the thickness of the layers
        :param vx_sz: voxel size in Armstrongs
        :param sigma: sigma for the gaussian filter
        :param fact: scalling factor (2 or 3)
        :return: the ndarray representing the bilayers after applying a gaussian filter
        """
    # Generating distance-to-layer-map
    ct_den = np.repeat(np.repeat(np.repeat(ct_den,fact,axis=0), fact, axis=1),fact,axis=2)

    ct_dist_in = np.float32(distance_transform_edt(ct_den))
    ct_dist_out = np.float32(distance_transform_edt(-ct_den + 1))
    ct_dist_in *= vx_sz
    #ct_dist_out[ct_dist_out>0] -= 1
    ct_dist_out *= vx_sz
    ct_dist = ct_dist_in - ct_dist_out
    del ct_dist_in
    del ct_dist_out

    # Taking the max value of the array
    ct_max_v = ct_dist.max()

    thr_o = -lyr_dist
    ct_lyr_o = np.copy(ct_dist)
    ct_lyr_o[(ct_lyr_o >= thr_o - lyr_thick) & (ct_lyr_o <= thr_o + lyr_thick)] = ct_max_v + 1
    ct_lyr_o[ct_lyr_o < ct_max_v] = 0
    ct_lyr_o[ct_lyr_o > 0] = 1

    thr_i = lyr_dist
    ct_lyr_i = np.copy(ct_dist)
    del ct_dist
    ct_lyr_i[(ct_lyr_i >= thr_i - lyr_thick) & (ct_lyr_i <= thr_i + lyr_thick)] = ct_max_v + 1
    ct_lyr_i[ct_lyr_i < ct_max_v] = 0
    ct_lyr_i[ct_lyr_i > 0] = 1

    ct_blyr = ct_lyr_o + ct_lyr_i
    #return ct_blyr
    ct_blyr = lin_map(density_norm(gaussian_filter(ct_blyr, sigma), inv=True), ub=0, lb=1)
    return ct_blyr

def reduce(ct_den, fact):
    """
        Reduce an MRC tomogram taking the aritmetic mean of every factxfactxfact block

        :param ct_den: the input density map
        :param fact: reducing factor
        """
    # Generating distance-to-layer-map
    return ct_den.reshape(100,fact,100,fact,100,fact).mean(axis=(1,3,5))

def gen_lyr(ct_den, lyr_thick, vx_sz, sigma):
    ct_dist_in = np.float32(distance_transform_edt(ct_den))
    ct_dist_out = np.float32(distance_transform_edt(-ct_den + 1))
    ct_dist_in *= vx_sz
    ct_dist_out *= vx_sz
    ct_dist = ct_dist_in - ct_dist_out
    del ct_dist_in
    del ct_dist_out

    # Taking the max value of the array
    ct_max_v = ct_dist.max()

    ct_dist[(ct_dist<=vx_sz) & (ct_dist>=-vx_sz)] = ct_max_v+1;
    ct_dist[ct_dist < ct_max_v] = 0
    ct_dist[ct_dist > 0] = 1

    # return ct_dist
    ct_lyr = lin_map(density_norm(gaussian_filter(ct_dist, sigma), inv=True), ub=0, lb=1)
    return ct_lyr

