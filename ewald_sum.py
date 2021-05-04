import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import integrate
from scipy.special import erfc
# from scipy.constants import c

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


# increase the size of the supercell
def supercell_vector(bravis, dim):
    supercell = bravis * dim
    return supercell
# Set lattice vectors in real space and reciprocal space
atom = basis[0]
# Get the area of the unit cell using cross product
A = np.linalg.norm(np.cross(bv[0], bv[1]))
# print("Unit cell area A = {}".format(A))
# Define some constant values
sigma = 1.15        # need testing, here we choose sigma = 5/ L
magnetic_moment = 4.548
c = 274.072
moment_vector = np.array([0, 0, 1])


# Ewald sum to get the Madelung constants, notice that the constant is in  1/m^3 unit
def madlung_constant(dim):
    m_constant = 0.0
    r_pos = util.setup_pbc(bv, atom, dim)
    rec = reciprocal(bv)
    # print("Reciprocal lattice vectors are :\n {}".format(rec))
    g_pos = util.setup_pbc(rec, atom, dim)
    # Calculate only the sqrt of x and y of the position vector
    r_vector_lengths = [np.linalg.norm(i - atom) for i in r_pos]    # in a_0 unit
    g_vector_lengths = [np.linalg.norm(j - atom) for j in g_pos]
    for r in r_vector_lengths:
        if r != 0:
            sum_1 = (erfc(r / (2 * sigma))) / (r ** 3)
            sum_2 = (np.exp(- r ** 2 / (4 * sigma ** 2))) / (sigma * math.sqrt(np.pi) * r ** 2)
            m_constant = m_constant + sum_1 + sum_2

    for g in g_vector_lengths:
        if g != 0:
            sum_3 = g * erfc(g * sigma)
            sum_4 = 1 / (sigma * math.sqrt(np.pi)) * np.exp(- g ** 2 * (sigma ** 2))
            m_constant = m_constant - (2 * np.pi / A) * (sum_3 - sum_4)

    m_constant = m_constant - 1 / (6 * (sigma ** 3) * math.sqrt(np.pi)) + 2 * math.sqrt(np.pi) / (A * sigma)
    return m_constant, r_vector_lengths, g_vector_lengths


def dipoler_energy(dim, length, madlung):
    energy = 0
    for i in range(len(length)):
        energy = energy + magnetic_moment ** 2 / (c ** 2) * madlung * np.dot(moment_vector, moment_vector)
    print("{} \t Madlung = {}\t E_dd = {}".format(dim[0], madlung, energy))
    return energy


for n in range(2, 150):
    dimension = [n, n, 1]
    M, r_lengths, g_lengths = madlung_constant(dimension)
    E_dd = dipoler_energy(dimension, r_lengths, M)
    plt.plot(n, E_dd, 'ob')
    # plt.plot(n, M, 'or')
plt.show()
