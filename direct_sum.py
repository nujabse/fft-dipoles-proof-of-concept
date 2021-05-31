import numpy as np
import concurrent.futures
import util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
import csv
import pymatgen.core as mg

"""
Rewrite periodical direct sum with classes
"""

# Get structure information from POSCAR
# TODO: Set magnetic atoms from user input
# get bravis lattice vector from PoSCAR
# get only the Mn atoms
direct_struct = mg.Structure.from_file("3u.c-POSCAR")
bv = direct_struct.lattice.matrix
print(bv)
Mn_coords = [a.coords for a in direct_struct if a.specie.symbol == 'Mn']
Mn_direct = [a.frac_coords for a in direct_struct if a.specie.symbol == 'Mn']
# Notice that it will return the cartesian coordinates of the atoms
basis_cartesian = np.asarray(Mn_coords)
basis_frac = np.asarray(Mn_direct)
# Here we are still using the old direct basis
basis = basis_frac
# Definition of bravais vectors and basis atoms
system = "Mn-Pt-2-Sz"

# Precision of printed output
np.set_printoptions(precision=10)
# As numpy.dot has problems with multiprocessing, we are here directly convert fractional coordinates (once and all)
basis = np.dot(basis, bv)


class Atom:
    """
    Object representation of spin, position
    """

    def __init__(self, position: np.ndarray, spin: np.ndarray):
        # fractional coordinates
        self.position = position
        self.spin = spin

    def __repr__(self) -> str:
        return 'Fractional coord({0.position}), spin({0.spin})'.format(self)

    def __eq__(self, other):
        if isinstance(other, Atom):
            return (np.array_equal(self.spin, other.spin)) and (np.array_equal(self.position, other.position))
        else:
            print("Can not compare")
            return NotImplemented


def supercell(basis_atoms: list, dimension: list, vectors: np.ndarray) -> list:
    lattice = []
    for i in range(-dimension[0], dimension[0] + 1):
        for j in range(-dimension[1], dimension[1] + 1):
            for center in basis_atoms:
                atom_position = center.position + i * vectors[0] + j * vectors[1]
                # Inherit the spin configuration from center atom
                atom_spin = center.spin
                atom = Atom(atom_position, atom_spin)
                lattice.append(atom)
    return lattice


class SuperCell:
    """
    Object representation of list of Atom objects
    """

    def __init__(self, lattice_basis: np.ndarray, lattice_vectors: np.ndarray, spin: np.ndarray, dimension: list,
                 layers=1,
                 afm_intra=False, afm_inter=True):
        self.basis = lattice_basis
        self.vectors = lattice_vectors
        self.spin = spin
        self.dimension = dimension
        self.layers = layers
        self.afm_inter = afm_inter  # Interlayer afm
        self.afm_intra = afm_intra  # Intralayer afm

    # TODO: Extract layer information from basis coordinates
    @property
    def basis_atoms(self) -> list:
        basis_atoms = []
        # Build basis atoms spin configurations on afm or fm conditions
        if self.afm_intra:
            if self.afm_inter:
                pass
            else:
                pass
        else:
            # The case in MnPt multilayer
            if self.afm_inter:
                repeats = len(self.basis) / self.layers
                basis_atoms = [Atom(a, (-1) ** (index % repeats) * self.spin) for index, a in
                               enumerate(self.basis, start=1)]
            else:
                basis_atoms = [Atom(a, self.spin) for a in self.basis]
        return basis_atoms

    @property
    def atoms(self) -> list:
        atoms = []
        atoms = supercell(self.basis_atoms, self.dimension, self.vectors)
        return atoms

    def plot_atoms(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for a in self.atoms:
            for b in self.basis_atoms:
                if a == b:
                    print(a, b)
                    if np.array_equal(a.spin, np.array([0, 0, 5])):
                        ax.scatter(a.position[0], a.position[1], a.position[2], s=35, marker='D', c='r')
                    else:
                        ax.scatter(a.position[0], a.position[1], a.position[2], s=35, marker='D', c='b')
                elif np.array_equal(a.spin, np.array([0, 0, 5])):
                    ax.scatter(a.position[0], a.position[1], a.position[2], c='r')
                elif np.array_equal(a.spin, np.array([0, 0, -5])):
                    ax.scatter(a.position[0], a.position[1], a.position[2], c='b')
        plt.show()

    def __repr__(self) -> str:
        return 'Basis: {0.basis} \n Bravais vectors: {0.vectors}'.format(self)


def dipolar_energy(center: Atom, lattice: list) -> float:
    e_dip = 0.0
    for i, a in enumerate(lattice):
        matrix = util.dipoleMatrix(a.position - center.position)
        e_dip += -0.5 * np.matmul(center.spin, np.matmul(matrix, a.spin))
    return e_dip


sc = SuperCell(basis, bv, np.array([0, 0, 5]), [2, 2, 1], layers=3)
sc.plot_atoms()
