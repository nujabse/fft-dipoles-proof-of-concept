import numpy as np

fac = 0.9274
NN = 6  # Number of nearest neighbours
box_length = 4.333799839  # Basic lattice box length


# return a 3-dim numpy array with the positons in the lattice
def setUpLattice(bv, N, basis=[[0, 0, 0]]):
    result = []
    for c in range(N[2]):
        for b in range(N[1]):
            for a in range(N[0]):
                for base in basis:
                    result.append(bv[0] * a + bv[1] * b + bv[2] * c + base)
    return np.array(result)


# Try to expand the lattice according to periodical boundary conditions
def setup_pbc(bv, atom, cell_dimension):
    result = []
    for i in range(-cell_dimension[0], cell_dimension[0] + 1):
        for j in range(-cell_dimension[1], cell_dimension[1] + 1):
            # convert array element to basic type
            # print("i = ", i, "j= ", j)
            if i == 0 and j == 0:
                continue
            else:
                super_cell_atom = atom + i * bv[0] + j * bv[1]
            result.append(super_cell_atom)
    return result


# builds spins on the lattice (lenght normalized to 1)
def buildSpins(lattice, config="Random"):
    if config == "Random":
        phi = np.random.rand(len(lattice))
        theta = np.random.rand(len(lattice))
        x = np.cos(phi * 2 * np.pi) * np.sin(theta * 2 * np.pi)
        y = np.sin(phi * 2 * np.pi) * np.sin(theta * 2 * np.pi)
        z = np.cos(theta * 2 * np.pi)
        result = np.dstack((x, y, z)).reshape(len(lattice), 3)
    elif config == "PlusZ":
        x = np.zeros(len(lattice))
        y = np.zeros(len(lattice))
        z = np.ones(len(lattice)) * 4.548
    elif config == "PlusX":
        x = np.ones(len(lattice)) * 4.548
        y = np.zeros(len(lattice))
        z = np.zeros(len(lattice))
    elif config == "PlusY":
        x = np.zeros(len(lattice))
        y = np.ones(len(lattice))
        z = np.zeros(len(lattice))
    result = np.dstack((x, y, z)).reshape(len(lattice), 3)
    return result


# dipole matrix -> for r=0 returns 0 matrix
def dipoleMatrix(r_vect):
    if r_vect[0] ** 2 + r_vect[1] ** 2 + r_vect[2] ** 2 > 1e-10:
        x = r_vect[0]
        y = r_vect[1]
        z = r_vect[2]
        r = np.sqrt(r_vect[0] ** 2 + r_vect[1] ** 2 + r_vect[2] ** 2)
        result = (1 / r ** 5) * fac * np.array(
            [[3 * x ** 2 - r ** 2, 3 * x * y, 3 * x * z],
             [3 * x * y, 3 * y ** 2 - r ** 2, 3 * y * z],
             [3 * x * z, 3 * y * z, 3 * z ** 2 - r ** 2]]
        )
    else:
        result = np.zeros(9).reshape(3, 3)
    return result


# Brute Force Gradients:
def calculateGradientsBF(lattice, spins):
    gradients = np.zeros(spins.shape)
    for k in range(len(lattice)):
        grad = np.zeros(3)
        for j in range(len(lattice)):
            if (j == k):  # skip self-interactions
                continue
            D = dipoleMatrix(lattice[k] - lattice[j])
            grad += - np.matmul(D, spins[j])
        gradients[k] = grad
    return gradients


# Brute Force the energy
def calculateEnergyBF(lattice, spins):
    E = 0
    for i in range(len(lattice)):
        for j in range(len(lattice)):
            if (i == j):
                continue
            D = dipoleMatrix(lattice[j] - lattice[i])
            # E_spin = -0.5 * np.matmul(spins[i], np.matmul(D, spins[j]))
            # print(
            #     "Spin {} coord {} and Spin {} cood {}, Matrix = {}, dipole = {}".format(i, lattice[i], j, lattice[j], D,
            #                                                                             E_spin))
            E += -0.5 * np.matmul(spins[i], np.matmul(D, spins[j]))
    return E


# Calculate dipole-dipole energy of the center atom
def calculate_energy_pbc(lattice, atom, spin, spins):
    E = 0
    for i in range(len(lattice)):
        D = dipoleMatrix(lattice[i] - atom)
        E += -0.5 * np.matmul(spin, np.matmul(D, spins[i]))
    return E


# calculate dipole-dipole energy on each seperate atom
def calculateEachEnergyBF(lattice, spins):
    E_DDI_BF_each = []
    for i in range(len(lattice)):
        E_DDI_atom = 0
        for j in range(len(lattice)):
            if (i == j):
                continue
            D = dipoleMatrix(lattice[j] - lattice[i])
            E_spin = -0.5 * np.matmul(spins[i], np.matmul(D, spins[j]))
            E_DDI_atom += E_spin
        E_DDI_BF_each.append(E_DDI_atom * 0.00425438)
    return E_DDI_BF_each


def SC_vectors(lattice_constant=1):
    return lattice_constant * np.array([[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 1]
                                        ]
                                       )


# converts 3D numpy array to row-major 1D array
def convertToNumpyStyle(x, N):
    new_shape = [N[2], N[1], N[0]]
    for i in range(1, len(x.shape)):
        new_shape.append(x.shape[i])
    result = x.reshape(new_shape)
    result = np.swapaxes(result, 0, 2)
    return result


# converts row-major 1D array to 3D numpy array
def convertToSpiritStyle(x):
    N = x.shape
    new_shape = [N[2] * N[1] * N[0]]
    for i in range(3, len(x.shape)):
        new_shape.append(x.shape[i])
    result = np.swapaxes(x, 0, 2)
    result = result.reshape(new_shape)
    return result


# sub is an array of row major 1D arrays where each 1D array represents a scalar \
# quantity living on a different sublattice
# joinSublattices joins these into one single 1D array where the quantities on different
# sublattices lie consecutively (spirit style)
def joinSublattices(sub, N):
    shape = [len(sub[0]) * len(sub)]
    for i in range(1, len(sub[0].shape)):
        shape.append(sub[0].shape[i])
    temp = np.zeros(shape)
    for c in range(N[2]):
        for b in range(N[1]):
            for a in range(N[0]):
                for i in range(len(sub)):
                    temp[i + a * len(sub) + b * len(sub) * N[0] + c * len(sub) * N[0] * N[1]] = sub[i][
                        a + b * N[0] + c * N[0] * N[1]]
    return temp
