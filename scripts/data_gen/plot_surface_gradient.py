import os

from sympy.codegen.ast import float32

from polnet.lio import load_mrc, write_mrc
import numpy as np

print(os.getcwd())
v = load_mrc(os.getcwd() + '/../../data'+ "/data_generated/all_v4/tomo_lbls_0.mrc")

v[v!=1] = 0
v = v.astype(np.float32)

z = np.indices(v.shape)[2]  # Creates a 3D grid of indices, where the third coordinate is z

# Apply the transformation (z - 125) / 250 to each element based on the z-coordinate
v[v==1] = v[v==1] + (z[v==1] - 125) / 250

write_mrc(v.astype(np.float32), os.getcwd() + '/../../data' +"/data_generated/all_v4/tomo_lbls_grad_0.mrc", v_size=10)

