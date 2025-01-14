import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import erfc
# from scipy.constants import c

import util

# Set lattice structure, notice that the lattice constant is in Angstrom units
# We will transform it into Rydeberg Atomic units in the calculation of energy
# MnBi2Te4 crystal structure
# bv = np.array([[4.3337998390000001, 0.0000000000000000, 0.0000000000000000],
#                [-2.1668999195000000, 3.7531807554999999, 0.0000000000000000],
#                [0.0000000000000000, 0.0000000000000000, 30.9099998474000017]]
#               )
# CrI3 crystal structure
bv = np.array([[6.718390000000000, 0.000000000000000, 0.000000000000000],
               [-3.359194999999999, 5.818296412531336, 0.000000000000000],
               [0.000000000000001, 0.000000000000002, 19.806999999999999]])

# MnPS3 crystal structure
# bv = np.array([[6.0700000000000003, 0.0000000000000000, 0.0000000000000000],
#               [-3.0350000000000001, 5.2567742009715426, 0.0000000000000000],
#               [-0.0000000000000001, 0.0000000000000001, 28.0000000000000000]])
# basis vectors in *direct coordinates*, will transform to cartesian coordinates in constructing lattice
# basis = np.array([[0.66667, 0.33333, 0.49617895]])  # basis position for MnBi2Te4
# basis = np.zeros((1, 3))     # Set the basis to the center of the lattice
basis = np.array([[0.333333999, 0.666666031, 0.499999970],
                  [0.666665971, 0.333333999, 0.499999970]])  # basis position for CrI3

# basis = np.array([[0.0000000000000000, 0.0000000000000000, 0.0960225520000009],
#                   [0.3333333333333357, 0.6666666666666643, 0.0960225520000009]]) # basis position for MnPS3
# Transform lattice units into Rydberg atomic units (a.u.)
bv = bv * 1.88973
# Set lattice vectors in real space and reciprocal space
rec = util.reciprocal(bv)
# Transform basis atom to cartesian coordinates
basis_cart_real = np.dot(basis, bv)
basis_cart_reciprocal = basis_cart_real
# basis_cart_reciprocal = np.dot(basis, rec)
basis_frac_rec = basis @ bv @ np.linalg.inv(rec)
# Precision of printed output
np.set_printoptions(precision=10)

# Get the area of the unit cell using cross product
A = np.linalg.norm(np.cross(bv[0], bv[1]))
# print("Unit cell area A = {}".format(A))
# Define some constant values
sigma = 0.8  # need testing, here we choose sigma = 5/ L
# magnetic_moment = 4.548  # Mn atom magnetization
magnetic_moment = 2.873  # Cr atom magnetization
mu_0 = 1.2566370614e-6
c = 274.072
# define a conversion matrix in spherical coordinates
mat = np.array([[-1 / 2, 0, 0], [0, -1 / 2, 0], [0, 0, 1]])
moment_vector = np.array([0, 0, 1])
# moment_vector = np.array([1, 0, 0])
sqpi = math.sqrt(np.pi)


# Ewald sum to get the Madelung constants, notice that the constant is in  1/m^3 unit
# mainly uses the formula described in
# https://doi.org/10.1103/physrevb.51.9552
def madlung_constant(dim, basis_atom, center_atom, basis_atom_rec, center_atom_rec, color='bo', text=None):
    m_constant = 0.0
    r_pos = util.setup_pbc(bv, center_atom, dim)
    # add center atom to lattice in case there are two atoms in unit cell
    if not np.array_equal(basis_atom, center_atom):
        r_pos.append(center_atom)
        # print("Center atom is not the basis atom")
    else:
        pass
        # print("Center atom is the basis atom")
    # util.plot_moment(r_pos, dim[0] - 1, center_atom, color=color, text=text)
    # print("Reciprocal lattice vectors are :\n {}".format(rec))
    g_pos = util.setup_pbc(rec, center_atom_rec, dim)
    # g_pos = [v @ bv @ np.linalg.inv(rec) for v in r_pos]
    if not np.array_equal(basis_atom_rec, center_atom_rec):
        g_pos.append(center_atom_rec)
        print("Center Reciprocal atom is not the basis atom")
    else:
        pass
        print("Center atom is the basis atom")
    # util.plot_moment(r_pos, dim[0] - 1, center_atom, color=color, text=text)
    util.plot_moment(g_pos, dim[0] - 1, center_atom_rec, color=color, text=text)
    # Calculate only the sqrt of x and y of the position vector
    r_vector_lengths = [np.linalg.norm(i - basis_atom) for i in r_pos]  # in a_0 unit
    g_vector_lengths = [np.linalg.norm(j - basis_atom_rec) for j in g_pos]
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


def dipoler_energy(loop, mom_v, madlung):
    # First consider relations between magnetic moment orientation and madelung constant
    madlung = madlung * np.linalg.multi_dot([mom_v, mat, mom_v])
    energy = magnetic_moment ** 2 / (c ** 2) * madlung
    # convert to meV unit
    energy = 13.6 * energy * 1000
    # print("{} \t Madlung = {}\t E_dd = {}".format(loop, madlung, energy))
    return energy


# Use the SI unit formula described in "10.1103/PhysRevB.88.144421"
# First define some auxiliary functions B and C
def B(r):
    out = (1 / r ** 3) * (erfc(sigma * r) + ((2 * sigma * r) / sqpi) * np.exp(- sigma ** 2 * r ** 2))
    return out


def C(r):
    out = (1 / r ** 5) * (3 * erfc(sigma * r) +
                          2 * sigma * r / sqpi * (3 + 2 * sigma ** 2 * r ** 2) * np.exp(-sigma ** 2 * r ** 2))
    return out


# Calculate the real part of the dipolar energy
def E_r(r, m_i, m_j):
    out = (1 / 2) * mu_0 / (4 * np.pi)


# Calculate the long range reciprocal part of the dipolar energy
def E_k(r):
    pass


# Calculate the self part of the dipolar energy. Here we omit the surface part contribution
def E_self(r):
    pass


# TODO: The currect implementation is false in  calculating CrI3 dipolar energy
super_cells = 6
for n in range(2, super_cells):
    dimension = [n, n, 1]
    M, r_lengths, g_lengths = madlung_constant(dimension, basis_cart_real[0], basis_cart_real[0],
                                               basis_cart_reciprocal[0], basis_cart_reciprocal[0], text='1')
    E_dd_plus = dipoler_energy(n, moment_vector, M)

    M, r_lengths, g_lengths = madlung_constant(dimension, basis_cart_real[0], basis_cart_real[1],
                                               basis_cart_reciprocal[0], basis_cart_reciprocal[1], color='ro', text='2')
    E_dd_minus = dipoler_energy(n, moment_vector, M)

    E_tot = E_dd_minus + E_dd_plus
    print("{}\tPlus Energy = {}\t Minus Energy = {} \t Total = {}".format(n, E_dd_plus, E_dd_minus, E_tot))

# plt.show()

#  start calculation
# fig, axs = plt.subplots(2, 1, constrained_layout=True)
# fig.suptitle('Magnetic dipolar energy of CrI3 ' + str(moment_vector), fontsize=16)
# for i in range(len(basis)):
#     atom = basis_cart_real[i]
#     atom_rec = basis_cart_reciprocal[i]
#     print("Calculating atom: {}".format(atom))
#     axs[i].set_xlabel('Supercell dimension')
#     axs[i].set_ylabel('Dipolar energy (meV)')
#     axs[i].set_title('Atom ' + str(i))
#     for n in range(2, 60):
#         dimension = [n, n, 1]
#         M, r_lengths, g_lengths = madlung_constant(dimension)
#         E_dd = dipoler_energy(n, moment_vector, M)
#         axs[i].plot(n, E_dd, 'ob')
#     print("Now plotting for atom {}".format(str(i)))

plt.show()
# plt.savefig(str(moment_vector) + " CrI3 " + str(dimension[0]) + "x" + str(dimension[1]) + ".pdf", dpi=300)
