import numpy as np
import util, mathematics
import matplotlib.pyplot as plt

# bravais vectors
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


# Test positions in supercell
# for atom in pos:
#     print(atom)

# Set up supercell with PBC
# for atom in pos:
#     pos_pbc = util.setup_pbc(bv, atom, N)
#     print("Atom: ", atom, "\n")
#     for i in range(len(pos_pbc)):
#         print("Neighbour: ", i, "Coord: ", pos_pbc[i])

# Try to plot the atoms in cartesian coordinates
# for atom in pos:
#     pos_pbc = util.setup_pbc(bv, atom, N)
#     for i in range(len(pos_pbc)):
#         print(pos_pbc[i][0], pos_pbc[i][1])
#         plt.plot(atom[0], atom[1], 'ro')
#         plt.plot(pos_pbc[i][0], pos_pbc[i][1], 'bo')
#         plt.text(pos_pbc[i][0], pos_pbc[i][1], str(i))
#     plt.show()

# Try to calculate the dipole dipole interaction strength of the center atom
# n = 1
# N = [n+1, n+1, 0]
# Here we only choose the one base atom
# Test for single case
# pos = util.setup_pbc(bv, basis[0], N)
# for i in range(len(pos)):
#     print("Neighbour: ", i, "Coord: ", pos[i])
# print("Atom: ", basis[0], "\n")
# for i in range(len(pos)):
#     print("Neighbour: ", i, "Coord: ", pos[i])
#     plt.plot(basis[0, 0], basis[0, 1], 'ro')
#     plt.plot(pos[i][0], pos[i][1], 'bo')
#     plt.text(pos[i][0], pos[i][1], str(i))
# # plt.show()
# name = n + 1
# plt.savefig(str(name) + "x" + str(name) + ".pdf", dpi=300)

# Construct function to plot magnetic moments configurations
def plotMoment(lattice, dimension):
    for i in range(len(lattice)):
        # print("Neighbour: ", i, "Coord: ", lattice[i])
        plt.plot(basis[0, 0], basis[0, 1], 'ro')
        plt.plot(lattice[i][0], lattice[i][1], 'bo')
        plt.text(lattice[i][0], lattice[i][1], str(i))
    dimension = dimension + 1
    plt.savefig(str(dimension) + "x" + str(dimension) + ".pdf", dpi=300)
    # Clear the figure, or it will stack onto the next
    plt.clf()


# Loop over many supercells with different sizes
for n in range(1, 500):
    N = [n + 1, n + 1, 0]
    pos = util.setup_pbc(bv, basis[0], N)
    # build spins
    spins = util.buildSpins(pos, "PlusZ")
    # for spin in spins:
    #     print(spin)
    # for i in range(len(pos)):
    #     print("Neighbour: ", i, "Coord: ", pos[i])
    # print("Atom: ", basis[0], "\n")
    E_dip = util.calculateEnergyBF(pos, spins)
    print("E_dip = ", E_dip)
    # plot the configuration into figure
    plotMoment(pos, n)
