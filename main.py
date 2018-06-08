import util
import numpy as np

bv1 = [1,0,0]
bv2 = [0,1,0]
bv3 = [0,0,1]
N = [5,5,1]

lattice = util.setUpLattice(np.array([bv1, bv2, bv3]), N)
spins=(util.buildSpins(lattice))

print(lattice)
print(spins)
print(util.dipoleMatrix([1,5,6]))