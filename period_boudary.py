import numpy as np
import concurrent.futures
import util, mathematics
import matplotlib.pyplot as plt
import csv

# bravais vectors
bv = np.array([[4.3337998390000001, 0.0000000000000000, 0.0000000000000000],
               [-2.1668999195000000, 3.7531807554999999, 0.0000000000000000],
               [0.0000000000000000, 0.0000000000000000, 30.9099998474000017]]
              )
# basis vectors
basis = np.array([[2.166922, 1.251048, 15.336891]])

# TODO consider multiple basis vectors

# number of basis cells in each direction
N = [2, 2, 1]
B = len(basis)

# Precision of printed output
np.set_printoptions(precision=8)

# Set up a system
pos = util.setUpLattice(bv, N, basis)
spins = util.buildSpins(pos, "PlusZ")


# Construct function to plot magnetic moments configurations
def plot_moment(lattice, dimension):
    for i in range(len(lattice)):
        # print("Neighbour: ", i, "Coord: ", lattice[i])
        plt.plot(basis[0, 0], basis[0, 1], 'ro')
        plt.plot(lattice[i][0], lattice[i][1], 'bo')
        plt.text(lattice[i][0], lattice[i][1], str(i))
    dimension = dimension + 1
    plt.savefig(str(dimension) + "x" + str(dimension) + ".pdf", dpi=300)
    # Clear the figure, or it will stack onto the next
    plt.clf()


# build spin for the center atom
spin = util.buildSpins(basis[0], "PlusY")[0]


# Loop over many supercells with different sizes
def calc_supercell_dipolar_energy(supercell):
    # for n in range(1, 1250):
    output = {}
    N = [supercell + 1, supercell + 1, 0]
    # build spins for the neighbouring atoms
    pos = util.setup_pbc(bv, basis[0], N)
    spins = util.buildSpins(pos, "PlusY")
    # build spins for neighbouring atoms
    E_dip = util.calculate_energy_pbc(pos, basis[0], spin, spins)
    # covert energy to meV unit
    E_dip = E_dip * 9.274009994e-24 / 2.1798723611035e-18 * 1e3 * 13.6
    print("System: ", str(supercell + 1) + 'x' + str(supercell + 1), "\tE_dip = ", E_dip)
    output["Loop"] = supercell + 1
    output["Energy"] = E_dip
    return output


# now use multi-processor to speed up the code
results = []
Number_of_cells = 1250
supercells = range(1, Number_of_cells)


def main():
    executor = concurrent.futures.ProcessPoolExecutor()
    for result in executor.map(calc_supercell_dipolar_energy, supercells):
        results.append(result)


# have to add this, or python will complain it can not sprawn new process
if __name__ == '__main__':
    main()

# calculate energy difference
diff = []
for i in range(-1, len(results) - 1):
    if i == 0:
        e_diff = results[0]["Energy"] - 0 # Here it may fail because i is an iterator
    else:
        e_diff = results[i + 1]["Energy"] - results[i]["Energy"]
    diff.append(e_diff)

# write output to csv file
with open('energy.csv', 'w', newline='') as file_handler:
    csv_writer = csv.writer(file_handler, delimiter=' ')
    for i in range(len(results)):
        csv_writer.writerow([results[i]["Loop"], results[i]["Energy"], diff[i]])

# plot the energy convergence plot
for i in range(len(results)):
    plt.plot(results[i]["Loop"], results[i]["Energy"], 'bo')
# plt.show()
plt.savefig("energy" + str(Number_of_cells + 1) + 'x' + str(Number_of_cells + 1) + ".pdf", dpi=300)
