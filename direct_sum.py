import numpy as np
import concurrent.futures
import util
import matplotlib.pyplot as plt
from functools import partial
import csv
import pymatgen.core as mg
import argparse

"""
Rewrite periodical direct sum with classes
"""

parser = argparse.ArgumentParser()
parser.add_argument('-f', required=True, dest='poscar', help='Specify the name of the POSCAR file', type=str)
parser.add_argument('-c', required=True, dest='center', help='Specify the center atom index (start from 0)', type=int)
parser.add_argument('-l', required=True, dest='layers', help='Input how many layers are the system', type=int)
parser.add_argument('-s', required=True, dest='system', help='Specify the out put file name', type=str)
parser.add_argument('-m', default=5, dest='magmom', help="Input the strength of the magnetic moment", type=int)
args = parser.parse_args()

poscar = args.poscar
center = args.center
layers = args.layers
system = args.system
magmom = args.magmom

# get only the Mn atoms
direct_struct = mg.Structure.from_file(poscar)
bv = direct_struct.lattice.matrix
Mn_coords = [a.coords for a in direct_struct if a.specie.symbol == 'Mn']
Mn_direct = [a.frac_coords for a in direct_struct if a.specie.symbol == 'Mn']
# Notice that it will return the cartesian coordinates of the atoms
basis_cartesian = np.asarray(Mn_coords)
basis_frac = np.asarray(Mn_direct)
# Here we are using the cartesian basis
basis = basis_cartesian

# Precision of printed output
np.set_printoptions(precision=10)


class Atom:
    """
    Object representation of spin, position
    """

    def __init__(self, position: np.ndarray, spin: np.ndarray):
        # cartesian coordinates
        self.position = position
        self.spin = spin

    def __repr__(self) -> str:
        return 'Cartesian coord({0.position}), spin({0.spin})'.format(self)

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

    def plot_atoms(self, basis_only=False) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if basis_only:
            for index, i in enumerate(self.basis_atoms, start=1):
                if np.array_equal(i.spin, s_z):
                    ax.scatter(i.position[0], i.position[1], i.position[2], s=35, marker='D', c='r')
                else:
                    ax.scatter(i.position[0], i.position[1], i.position[2], s=35, marker='D', c='b')
                ax.text(i.position[0], i.position[1], i.position[2], str(index))
            plt.tight_layout()
            plt.savefig('Output/Pt-interface/P-up/' + str(self.layers) + '-layers' + '.png', transparent=True, dpi=300)
        else:
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
    energy = 0.0
    for i, a in enumerate(lattice):
        matrix = util.dipoleMatrix(a.position - center.position)
        energy += -0.5 * np.matmul(center.spin, np.matmul(matrix, a.spin))
    # covert energy to meV unit
    energy = energy * 9.274009994e-24 / 2.1798723611035e-18 * 1e3 * 13.6
    return energy


supercell_limit = 200
# Here we are starting from calculating supercell 2x2x1
supercells = [[n, n, 1] for n in range(2, supercell_limit)]
e_dip_sz = []
e_dip_sx = []
s_z = np.array([0, 0, 5])
s_x = np.array([5, 0, 0])


# Auxiliary function for multiprocessing (one parameter only)
def dipolar_energy_supercell(spin: np.ndarray, dimension: list) -> float:
    super_cell = SuperCell(basis, bv, spin, dimension, layers=layers)
    # Notice that you need to change center atom as the layers become thicker
    energy = dipolar_energy(super_cell.basis_atoms[center], super_cell.atoms)
    # print(super_cell.basis_atoms[0])
    print('Supercell: {} \t E_dip: {}'.format(dimension, energy))
    return energy


dipolar_energy_sz = partial(dipolar_energy_supercell, s_z)
dipolar_energy_sx = partial(dipolar_energy_supercell, s_x)


# TODO: Use multithreading to control calculating multiple systems while using multiprocessing to speed up running on
# one task
def main():
    executor = concurrent.futures.ProcessPoolExecutor()
    for e_sz in executor.map(dipolar_energy_sz, supercells):
        e_dip_sz.append(e_sz)
    for e_sx in executor.map(dipolar_energy_sx, supercells):
        e_dip_sx.append(e_sx)


def delta(datalist: list) -> list:
    """
    Helper function to calculate difference of energy after each step
    """
    diff = []
    for index, data in enumerate(datalist, start=2):
        if index == 2:
            e_diff = data
        else:
            e_diff = data - datalist[index - 3]
        diff.append(e_diff)
    return diff


def data_writer(data_list: list, diff_list: list, name: str) -> None:
    with open('Output/Pt-interface/P-up/' + name + '-' + system + '-' + str(center) + '-' + str(supercell_limit) + 'x' + str(
            supercell_limit) + '-energy.csv', 'w',
              newline='') as file_handler:
        csv_writer = csv.writer(file_handler, delimiter=' ')
        for index, (data, difference) in enumerate(zip(data_list, diff_list), start=2):
            csv_writer.writerow([index, data, difference])


if __name__ == '__main__':
    main()
    print(system, layers)
    print("Dipolar Energy Sz = {}".format(e_dip_sz[-1]))
    print("Dipolar Energy Sx = {}".format(e_dip_sx[-1]))
    delta_sx = delta(e_dip_sx)
    delta_sz = delta(e_dip_sz)
    data_writer(e_dip_sx, delta_sx, 'Sx')
    data_writer(e_dip_sz, delta_sz, 'Sz')
    # Plot results
    plt.plot(range(2, supercell_limit), e_dip_sx, 'r')
    plt.plot(range(2, supercell_limit), e_dip_sz, 'b')
    plt.show()

# sc = SuperCell(basis, bv, np.array([0, 0, 5]), [2, 2, 1], layers=6)
# sc.plot_atoms(basis_only=True)
# e_dip = dipolar_energy(sc.basis_atoms[2], sc.atoms)
# print(len(sc.basis_atoms))
# print(e_dip)
