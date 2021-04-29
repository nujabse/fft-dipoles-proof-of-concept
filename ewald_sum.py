import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import integrate
from scipy.special import erfc
from scipy.constants import c

import util

np.set_printoptions(precision=8)
# Set lattice structure
bv = np.array([[4.3337998390000001, 0.0000000000000000, 0.0000000000000000],
               [-2.1668999195000000, 3.7531807554999999, 0.0000000000000000],
               [0.0000000000000000, 0.0000000000000000, 30.9099998474000017]]
              )
# basis vectors in *direct coordinates*, will transform to cartesian coordinates in constructing lattice
# basis = np.array([[0.66667, 0.33333, 0.49617895]])
basis = np.zeros((1, 3))     # Set the basis to the center of the lattice
# number of basis cells in each direction
N = [90, 90, 1]
B = len(basis)

# Precision of printed output
np.set_printoptions(precision=8)


# Get the reciprocal lattice vector, see Kitchens "surfaces.pdf", P. 39
# Also see the definition from "Introduction to solid state physics", P. 23
def reciprocal(lattice_vector):
    Astar = 2 * np.pi * np.linalg.inv(lattice_vector).T
    return Astar


# Test the reciprocal lattice vector formula, notice we add  2 pi constant here
# for i in range(3):
#     for j in range(3):
#         result = np.dot(bv[i], rec[j])
#         print("The dot product between {} and {} is {}\n".format(i, j, result))

# Set lattice vectors in real space and reciprocal space
atom = basis[0]
r_pos = util.setup_pbc(bv, atom, N)
rec = reciprocal(bv)
print("Reciprocal lattice vectors are :\n {}".format(rec))
g_pos = util.setup_pbc(rec, atom, N)
# Print out the lattice vectors in real space and reciprocal spaces
# for i in range(len(r_pos)):
#     print("Real space vector {} is: {}".format(i, r_pos[i]))
#     print("Reciprocal vector {} is: {}".format(i, g_pos[i]))
#     # Test the product of Real and Reciprocal vectors
#     print("Dot product of the two vectors is {}".format(np.dot(r_pos[i], g_pos[i])))
# Ewald sum to get the Madelung constants
M = 0.0
# Define some constant values
sigma = 2        #need testing
# Get the area of the unit cell using cross product
A = np.linalg.norm(np.cross(bv[0], bv[1]))
print("Unit cell area A = {}".format(A))
# lists of lattice vectors in real space and reciprocal space
# Note we are using Rydberg unit of length
r_lengths = [np.linalg.norm(i) for i in r_pos]    # in a_0 unit
g_lengths = [np.linalg.norm(j) for j in g_pos]

for r_length in r_lengths:
    if r_length != 0:
        sum_1 = (erfc(r_length / (2 * sigma))) / (r_length ** 3)
        sum_2 = (np.exp(- r_length ** 2) / (4 * sigma ** 2)) / (sigma * math.sqrt(np.pi) * r_length ** 2)
        M = M + sum_1 + sum_2

for g_length in g_lengths:
    if g_length != 0:
        sum_3 = g_length * erfc(g_length * sigma)
        sum_4 = 1 / (sigma * math.sqrt(np.pi)) * np.exp(- g_length ** 2 * (sigma ** 2))
        M = M - (2 * np.pi / A) * (sum_3 + sum_4)

M = M - 1 / (6 * (sigma ** 3) * math.sqrt(np.pi)) + 2 * math.sqrt(np.pi) / (A * sigma)
print("M = {}".format(M))

# Calculate the dipole-dipole energy
# Note that we are working in Rydberg units here
# Define some constants
# mu_b = math.sqrt(2)
magnetic_moment = 4.548
moment_vector = np.array([0, 0, 1])
E_dd = 0

for i in r_pos:
    E_dd = E_dd + magnetic_moment ** 2 / (c ** 2) * M * np.dot(moment_vector, moment_vector)
    E_dd = E_dd

print("Dipole energy for the central atom is: {}".format(E_dd))
