import numpy as np
import concurrent.futures
import util
import matplotlib.pyplot as plt
from functools import partial
import csv
import pymatgen.core as mg


# TODO: Set magnetic atoms from user input
# get bravis lattice vector from PoSCAR
# get only the Mn atoms
direct_struct = mg.Structure.from_file("POSCAR")
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
# system = "MnPS3-Sz "
# system = "MnPS3-Sx "
system = "MnPSe3-Sz "

# Precision of printed output
np.set_printoptions(precision=10)
# As numpy.dot has problems with multiprocessing, we are here directly convert fractional coordinates (once and all)
basis = np.dot(basis, bv)

# TODO: use user input or external file as lattice and basis parameter


# Loop over many supercells with different sizes
def calc_supercell_dipolar_energy(basis_atom, center_atom, dimension, atom_moment="PlusZ", lattice_moment="PlusZ"):
    output = {}
    N = dimension[0]
    # build spin for the center atom
    spin = util.buildSpins(basis_atom, atom_moment)[0]
    # build periodical lattice for the neighbouring atoms
    pos = util.setup_pbc(bv, center_atom, dimension)
    # add center atom to lattice in case there are two atoms in unit cell
    if not np.array_equal(basis_atom, center_atom):
        pos.append(center_atom)
        # print("Center atom is not the basis atom")
    else:
        pass
        # print("Center atom is the basis atom")
    # pos = util.setup_pbc_multiple_basis(bv, basis[0], basis, N)
    spins = util.buildSpins(pos, lattice_moment)
    # build spins for neighbouring atoms
    e_dip = util.calculate_energy_pbc(pos, basis_atom, spin, spins)
    # covert energy to meV unit
    e_dip = e_dip * 9.274009994e-24 / 2.1798723611035e-18 * 1e3 * 13.6
    print("System: ", str(N) + 'x' + str(N), "\tE_dip = ", e_dip)
    output["Loop"] = N
    output["Energy"] = e_dip
    return output


# Test the supercell configuration
# dim = [50, 50, 1]
# pos_plus = util.setup_pbc(bv, basis[0], dim)
# pos_minus = util.setup_pbc(bv, basis[1], dim)
# plot out the supercell
# util.plot_moment(pos_plus, dim[0], basis[0],  color='ro', text='↓')
# util.plot_moment(pos_minus, dim[0], basis[1], color='bo', text=r'$\uparrow$')
# Calculate out-of-plane energy
# dip_plus = calc_supercell_dipolar_energy(basis[0], basis[0], dim, lattice_moment="PlusZ")
# dip_minus = calc_supercell_dipolar_energy(basis[0], basis[1], dim, lattice_moment="-PlusZ")
# In-plane dipolar energy
# dip_plus = calc_supercell_dipolar_energy(basis[0], basis[0], dim, atom_moment="PlusX", lattice_moment="PlusX")
# dip_minus = calc_supercell_dipolar_energy(basis[0], basis[1], dim, atom_moment="PlusX", lattice_moment="-PlusX")
# print("+ : {} \t - : {}".format(dip_plus["Energy"], dip_minus["Energy"]))
# total_energy = dip_plus["Energy"] + dip_minus["Energy"]
# print("PlusZ dipole energy is : {}".format(total_energy))
# print("PlusX dipole energy is : {}".format(total_energy))
# Print out positions
# for i in range(len(pos_plus)):
#     print("+ : {} \t - : {}".format(pos_plus[i], pos_minus[i]))

# plt.show()
# plt.savefig(str(dim[0]) + "x" + str(dim[1]) + ".pdf", dpi=300)

# now use multi-processor to speed up the code
Number_of_cells = 200
supercells = [[n, n, 1] for n in range(2, Number_of_cells)]
results_plus = []
results_minus = []
results = []


def main():
    # Split tasks to subtasks
    calc_dip_z_parallel = partial(calc_supercell_dipolar_energy, basis[0], basis[0],
                                  atom_moment="PlusZ", lattice_moment="PlusZ")
    calc_dip_z_anti = partial(calc_supercell_dipolar_energy, basis[0], basis[1],
                              atom_moment="PlusZ", lattice_moment="-PlusZ")
    # calc_dip_x_parallel = partial(calc_supercell_dipolar_energy, basis[0], basis[0],
    #                               atom_moment="PlusX", lattice_moment="PlusX")
    # calc_dip_x_anti = partial(calc_supercell_dipolar_energy, basis[0], basis[1],
    #                           atom_moment="PlusX", lattice_moment="-PlusX")
    executor = concurrent.futures.ProcessPoolExecutor()
    for result_plus in executor.map(calc_dip_z_parallel, supercells):
        results_plus.append(result_plus)
    for result_minus in executor.map(calc_dip_z_anti, supercells):
        results_minus.append(result_minus)
    # for result_plus in executor.map(calc_dip_x_parallel, supercells):
    #     results_plus.append(result_plus)
    # for result_minus in executor.map(calc_dip_x_anti, supercells):
    #     results_minus.append(result_minus)
    # Add the two results together
    for _ in range(0, Number_of_cells - 2):
        out = {"Loop": _, "Energy": results_plus[_]["Energy"] + results_minus[_]["Energy"]}
        results.append(out)


# have to add this, or python will complain it can not sprawn new process
if __name__ == '__main__':
    main()
    # Print the final energy
    print("System: {} Total dipolar energy = {}".format(Number_of_cells, results[-1]["Energy"]))

    # calculate energy difference
    diff = []
    for i in range(-1, len(results) - 1):
        if i == 0:
            e_diff = results[0]["Energy"] - 0  # Here it may fail because i is an iterator
        else:
            e_diff = results[i + 1]["Energy"] - results[i]["Energy"]
        diff.append(e_diff)

    # write output to csv file
    with open(system + str(Number_of_cells) + 'x' + str(Number_of_cells) + '-energy.csv', 'w', newline='') \
            as file_handler:
        csv_writer = csv.writer(file_handler, delimiter=' ')
        for _ in range(len(results)):
            csv_writer.writerow([results[_]["Loop"], results[_]["Energy"], diff[_]])

    # plot the energy convergence plot
    # plt.clf()
    # Use list comprehension to avoid variable leaking
    plt.plot([x for x in range(2, Number_of_cells)], [e["Energy"] for e in results], 'bo', markersize=3)
    # need to manually set y limit in plotting Sx
    ax = plt.gca()
    ax.set_ylim(0.015, 0.0151)
    plt.xlabel("Number of supercells")
    plt.ylabel("Dipolar Energy (meV)")
    plt.tight_layout()  # Need this, or plot will run out of figure
    plt.savefig(system + str(Number_of_cells) + 'x' + str(Number_of_cells) + ".pdf", dpi=300)

    # plot the energy difference plot
    plt.clf()
    # plt.yscale("log")
    plt.semilogy([x for x in range(2, Number_of_cells)], [d for d in diff], 'ro', markersize=3)
    plt.xlabel("Number of supercells")
    plt.ylabel("Energy difference")
    plt.tight_layout()  # make plot tighter
    plt.savefig(system + "Energy difference " + str(Number_of_cells) + 'x' + str(Number_of_cells) + ".pdf", dpi=300)
