"""
Script for processing images generated by curvatubes
    Input:
        - nii.gz source file
    Output:
        - The simulated density maps
        - The 3D reconstructed tomograms
        - Micrograph stacks
        - Polydata files
        - STAR file mapping particle coordinates and orientations with tomograms
"""

__author__ = 'Juan Diego Gallego Nicolas'

import sys
import csv
import time
import random

import numpy as np
import nibabel as nib

from polnet.lio import load_mrc, write_mrc
from polnet.membrane.membrane import MbEllipsoid, MbSphere, MbCurvatubes
from polnet.utils import *


# Common tomogram settings
ROOT_PATH = os.path.realpath(os.getcwd() + '/../../data')

e = MbCurvatubes(tomo_shape=(100,100,100), layer_s=1.5, v_size=10, thick=30)
tomo = e.get_tomo()
tomo = np.float32(tomo)
write_mrc(tomo=tomo,v_size=1,fname=ROOT_PATH+"/tomo.mrc")
mask = e.get_mask()
mask = np.float32(mask)
write_mrc(tomo=mask,v_size=1,fname=ROOT_PATH+"/mask.mrc")

"""
e = MbSphere(tomo_shape=(100,100,100), center=(500,500,500), rad=300, layer_s=1.5, v_size=10, thick=30)
tomo = e.get_tomo()
tomo = np.float32(tomo)
write_mrc(tomo=tomo,v_size=1,fname=ROOT_PATH+"/tomo.mrc")
mask = e.get_mask()
mask = np.float32(mask)
write_mrc(tomo=mask,v_size=1,fname=ROOT_PATH+"/mask.mrc")
"""



