import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import integrate
from scipy.special import erf
from scipy.constants import c

import util


np.set_printoptions(precision=8)
# Set lattice structure
bv = np.array([[4.3337998390000001, 0.0000000000000000, 0.0000000000000000],
               [-2.1668999195000000, 3.7531807554999999, 0.0000000000000000],
               [0.0000000000000000, 0.0000000000000000, 30.9099998474000017]]
              )
# basis vectors
basis = np.array([[2.166922, 1.251048, 15.336891]])
# number of basis cells in each direction
N = [2, 2, 1]
B = len(basis)

# Precision of printed output
np.set_printoptions(precision=8)

# Set up a system
pos = util.setUpLattice(bv, N, basis)
spins = util.buildSpins(pos, "PlusZ")

# Get the reciprocal lattice vector
def reciprocal(lattice_vector):
    Astar = np.linalg.inv(lattice_vector).T
    gstar = np.dot(Astar, Astar.T)
    return gstar

# Ewald sum to get the Madelung constants
M = 0.0
# Define some constant values
sigma = 2  # need testing
# Get the area of the unit cell using cross product
A = np.linalg.norm(np.cross(bv[0], bv[1]))
print("Unit cell area A = {}".format(A))
# lists of lattice vectors in real space and reciprocal space
r_lengths = []
g_lengths = []

for r_length in r_lengths:
    if r_length != 0:
        sum_1 = (erf(r_length / (2 * sigma))) /  (r_length ** 3)
        sum_2 = (np.exp(- r_length ** 2) / (4 * sigma ** 2)) / (sigma * math.sqrt(np.pi) * r_length ** 2)
        M = M + sum_1 + sum_2

for g_length in g_lengths:
    if g_length != 0:
        sum_3 = g_length * erf(g_length * sigma)
        sum_4 = 1 / (sigma * math.sqrt(np.pi)) * np.exp(- g_length ** 2 * (sigma ** 2))
        M = M - (2 * np.pi / A) * (sum_3 + sum_4)

M = M - 1 / (6 * (sigma ** 3) * math.sqrt(np.pi)) + 2 * math.sqrt(np.pi) / (A * sigma)
print("M = {}".format(M))

# Calculate the dipole-dipole energy

moments = []
atom = 1
E_dd = 0

for i in moments:
   E_dd = E_dd + atom * i / (c ** 2) * M
