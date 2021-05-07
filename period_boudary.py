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
basis = np.array([[0.66667, 0.33333, 0.49617895]])  # basis position for MnBi2Te4 (in fractional coordinates)
# As numpy.dot has problems with multiprocessing, we are here directly convert fractional coordinates (once and all)
basis = np.dot(basis, bv)
# basis = np.array([[2.166922, 1.251048, 15.336891]]) # basis position for MnBi2Te4 (in cartesian coordinates)

# TODO consider multiple basis vectors
# Precision of printed output
np.set_printoptions(precision=8)

# build spin for the center atom
spin = util.buildSpins(basis[0], "PlusY")[0]


# Loop over many supercells with different sizes
def calc_supercell_dipolar_energy(supercell):
    # for n in range(1, 1250):
    output = {}
    N = [supercell, supercell, 0]
    # build spins for the neighbouring atoms
    pos = util.setup_pbc(bv, basis[0], N)
    spins = util.buildSpins(pos, "PlusY")
    # build spins for neighbouring atoms
    E_dip = util.calculate_energy_pbc(pos, basis[0], spin, spins)
    # covert energy to meV unit
    E_dip = E_dip * 9.274009994e-24 / 2.1798723611035e-18 * 1e3 * 13.6
    print("System: ", str(supercell) + 'x' + str(supercell), "\tE_dip = ", E_dip)
    output["Loop"] = supercell
    output["Energy"] = E_dip
    return output


# now use multi-processor to speed up the code
results = []
Number_of_cells = 200
supercells = range(2, Number_of_cells)


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
plt.savefig("energy" + str(Number_of_cells) + 'x' + str(Number_of_cells) + ".pdf", dpi=300)
