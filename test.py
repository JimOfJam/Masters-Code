import gaussian as g
import numpy as np
import timeit 
import matplotlib.pyplot as plt 

j = 0+1j 

# This function will help keep our indices notationally consistent with the Wikipedia page for the Hubbard model
def index(i,sigma):
    return 2*(i%4)+sigma 

# Initialising the Hilbert space
space = g.FermionicSpace(8)

# Defining the Hamiltonian 
space.hamiltonian = np.zeros((space.dimension,space.dimension),dtype="complex_")
for i in range(4):
    for sigma in range(2):
        firstterm = np.matmul(space.raising[index(i,sigma)],space.lowering[index(i+1,sigma)])
        space.hamiltonian -= firstterm + g.adjoint(firstterm)

def vary(psi,epsilon):
    psi = psi + epsilon*space.randomOrthogonal(psi)
    return psi/np.linalg.norm(psi)


# Search for the ground state 
tolerance = 0.01; epsilon = 0.1;variation = 500
psi = space.randomState()
expectationValue = lambda x : g.prodN([g.adjoint(x),space.hamiltonian,x])
energy = expectationValue(psi)
lastUpdate = 0
energies = []

start = timeit.default_timer()
for i in range(100000):
    candidate = vary(psi,epsilon)
    candidateEnergy = expectationValue(candidate)
    if candidateEnergy.real < energy:
        psi = candidate
        energy = candidateEnergy.real
        with open('energy.txt','w') as f:
            f.writelines(str(i) + ": " + str(energy)+"\n")
        lastUpdate = i 
    elif i-lastUpdate == variation:
        epsilon = epsilon*0.99
    energies.append(energy)

stop = timeit.default_timer()
print("Method terminated in %s seconds." % (stop-start,))
print(energy,epsilon)


plt.plot(range(i+1),energies)
plt.show()