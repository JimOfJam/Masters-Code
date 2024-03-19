import numpy as np
import gaussian as g
import scipy.linalg
import matplotlib.pyplot as plt 
import timeit 

i = 0+1j
n = 100

def index(j,sigma,k):
    return int(j % (n/2) + sigma*(n/2) + k*n)

def testIndex(i,sigma):
    return i%4+sigma*4


h = np.zeros((2*n,2*n))
for j in range(int(n/2)):
    for sigma in range(2):
        h[index(j,sigma,0)][index(j+1,sigma,1)] = -1
        h[index(j,sigma,1)][index(j+1,sigma,0)] = 1
        h[index(j+1,sigma,0)][index(j,sigma,1)] = -1
        h[index(j+1,sigma,1)][index(j,sigma,0)] = 1

h = 0.5*(h - h.T)
print(h)

def expectationValue(Omega):
    return -0.25*np.trace(np.matmul(h,Omega))

def vary(Omega,epsilon):
    K = np.random.rand(2*n,2*n)
    K = K - K.T 
    return g.prodN([scipy.linalg.expm(epsilon*K),Omega,scipy.linalg.expm(epsilon*K.T)])

Omega = np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
energy = expectationValue(Omega)
energies = []
epsilon = 0.01
lastUpdate = 0
adjustEpsilon = False

starttime = timeit.default_timer()
for j in range(1000):
    candidate = vary(Omega,epsilon)
    candidateEnergy = expectationValue(candidate)
    if candidateEnergy.real < energy:
        Omega = candidate
        energy = candidateEnergy.real 
        lastUpdate = j
    elif j - lastUpdate == 10 and adjustEpsilon == True:
        epsilon = epsilon*0.9
        lastUpdate = j
    energies.append(energy)
stoptime = timeit.default_timer()

print(j+1,"iterations terminated with a runtime of",stoptime-starttime,"seconds.")
print("Ground state energy:",energy)
print("Final value of epsilon:",epsilon)

plt.plot(range(j+1),energies)
plt.show()












"""space = g.FermionicSpace(n)

hamiltonian = np.zeros((2**n,2**n),dtype="complex_")
for j in range(2*n):
    for k in range(2*n):
        hamiltonian += 0.5*i*h[j][k]*np.matmul(space.majorana[j],space.majorana[k])

space.hamiltonian = np.zeros((space.dimension,space.dimension),dtype="complex_")
for j in range(4):
    for sigma in range(2):
        firstterm = np.matmul(space.raising[testIndex(j,sigma)],space.lowering[testIndex(j+1,sigma)])
        space.hamiltonian -= firstterm + g.adjoint(firstterm)

print(np.equal(hamiltonian,space.hamiltonian))"""