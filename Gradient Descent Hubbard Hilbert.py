
import gaussian as g
import numpy as np
import timeit 
import matplotlib.pyplot as plt 
import time

j = 0+1j 

# This function will help keep our indices notationally consistent with the Wikipedia page for the Hubbard model
def index(i,sigma):
    return i%4+sigma*4

n = 10

# Initialising the Hilbert space
space = g.FermionicSpace(n)

# Defining the Hamiltonian 
space.hamiltonian = np.zeros((space.dimension,space.dimension),dtype="complex_")
for i in range(int(n/2)):
    for sigma in range(2):
        firstterm = np.matmul(space.raising[index(i,sigma)],space.lowering[index(i+1,sigma)])
        space.hamiltonian -= firstterm + g.adjoint(firstterm)

# This is the function to be minimised 
expectationValue = lambda x : g.prodN([g.adjoint(x),space.hamiltonian,x])

# Finds the real ground state by computing eigenvalues directly
eig = np.linalg.eig(space.hamiltonian)
min = 0
for i in range(len(eig[0])):
    if eig[0][i].real < eig[0][min].real:
        min = i
print("Real ground state energy:",eig[0][min].real)

# Define the basis vectors of the Hilbert space
basis = []
for i in range(space.dimension):
    v = np.zeros((space.dimension,1))
    v[i] = 1
    basis.append(v)

# Perform the method of steepest descent
psi = space.randomState()
energy = expectationValue(psi).real
epsilon = 0.1

energies = []
starttime = timeit.default_timer()

for i in range(1000):
    gradient = np.array([g.prodN([g.adjoint(basis[k]),space.hamiltonian,psi])[0] for k in range(space.dimension)])
    candidate = psi - epsilon*gradient 
    candidate = candidate/np.linalg.norm(candidate)
    candidateEnergy = expectationValue(candidate).real
    if candidateEnergy < energy:
        psi = candidate
        energy = candidateEnergy 
    else:
        epsilon = epsilon*0.5
        if epsilon < 10**(-10):
            break
    
    if i % 10 == 0:
        print("Progress:" + str(i/10) + "%",end="\r") 
    energies.append(energy)

stoptime = timeit.default_timer()
print("Method terminated in",stoptime-starttime,"seconds with final epsilon value",str(epsilon)+".")
print("Ground state energy:",energy)

plt.plot(range(i+1),energies)
plt.show()
