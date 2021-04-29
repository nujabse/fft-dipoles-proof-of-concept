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
# Precision of printed output
np.set_printoptions(precision=8)


# Get the reciprocal lattice vector, see Kitchens "surfaces.pdf", P. 39
# Also see the definition from "Introduction to solid state physics", P. 23
def reciprocal(lattice_vector):
    Astar = 2 * np.pi * np.linalg.inv(lattice_vector).T
    return Astar

# Set lattice vectors in real space and reciprocal space
atom = basis[0]
# Get the area of the unit cell using cross product
A = np.linalg.norm(np.cross(bv[0], bv[1]))
# print("Unit cell area A = {}".format(A))
# Define some constant values
sigma = 1.15        #need testing, here we choose sigma = 5/ L
magnetic_moment = 4.548
moment_vector = np.array([0, 0, 1])
E_dd = 0


# Ewald sum to get the Madelung constants
def madlung_constant(dimension):
    M = 0.0
    r_pos = util.setup_pbc(bv, atom, dimension)
    rec = reciprocal(bv)
    # print("Reciprocal lattice vectors are :\n {}".format(rec))
    g_pos = util.setup_pbc(rec, atom, dimension)
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
    return M, r_lengths, g_lengths


def dipoler_energy(dimension, length, madlung):
    E_dd = 0
    for i in length:
        E_dd = E_dd + magnetic_moment ** 2 / (c ** 2) * madlung * np.dot(moment_vector, moment_vector)
        E_dd = E_dd / len(length)
    print("{} \t Madlung = {}\t E_dd = {}".format(dimension[0], madlung, E_dd))
    return E_dd


for n in range(2, 100):
    dimension = [n, n, 1]
    M, r_lengths, g_lengths = madlung_constant(dimension)
    E_dd = dipoler_energy(dimension, r_lengths, M)
    plt.plot(n, E_dd, 'ob')
    # plt.plot(n, M, 'or')
plt.show()
