import numpy as np
import gaussian as g
import scipy.linalg
import matplotlib.pyplot as plt 
import timeit 

i = 0+1j
n = 50

# Helps us index the Majorana modes correctly
def index(j,sigma,k):
    return int(j % (n/2) + sigma*(n/2) + k*n)

# The h_{ab} matrix which contracts with the Majorana modes to produce the Hamiltonian
h = np.zeros((2*n,2*n))
for j in range(int(n/2)):
    for sigma in range(2):
        h[index(j,sigma,0)][index(j+1,sigma,1)] = -1
        h[index(j,sigma,1)][index(j+1,sigma,0)] = 1
        h[index(j+1,sigma,0)][index(j,sigma,1)] = -1
        h[index(j+1,sigma,1)][index(j,sigma,0)] = 1
h = 0.5*(h - h.T)


# Define the basis elements of the Lie algebra 
basis = []
lieDim = n*(2*n-1)
for j in range(lieDim):
    entries = np.zeros((lieDim))
    entries[j] = 1
    basis.append(entries)

def RtoT(vector):
    index = 0
    K = np.zeros((2*n,2*n))
    for j in range(2*n):
        for k in range(j+1,2*n):
            K[j][k] = vector[index]
            K[k][j] = -vector[index]
            index += 1
    return K 

def lieProduct(A,B):
    return np.trace(np.matmul(A,B.T))

basisK = [RtoT(v) for v in basis]
for v in basisK:
    v = v/np.sqrt(lieProduct(v,v))





# This is the function to be optimised 
def expectationValue(x):
    return -0.25*np.trace(np.matmul(h,x))

def gradient(x):
    grad = np.zeros(lieDim)
    for j in range(lieDim):
        grad[j] = expectationValue(np.matmul(basisK[j],x)+np.matmul(x,basisK[j].T))
    return grad

def vary(x,K,epsilon):
    return g.prodN([scipy.linalg.expm(epsilon*K),x,scipy.linalg.expm(epsilon*(K.T))])

Omega = np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
K = np.random.rand(lieDim)
Omega = vary(Omega,RtoT(K),100)
energy = expectationValue(Omega).real
print("Initial energy: ",energy)
epsilon = 10

energies = []


starttime = timeit.default_timer()
for j in range(50):
    grad = gradient(Omega)
    candidate = vary(Omega,RtoT(grad),-epsilon)
    candidateEnergy = expectationValue(candidate).real

    if candidateEnergy < energy:
        Omega = candidate
        energy = candidateEnergy 
    elif epsilon < 10**(-10):
        print("Epsilon reached threshold after",j+1,"iterations.")
        break 
    else:
        epsilon = epsilon*0.5

    energies.append(energy)
stoptime = timeit.default_timer()

print("Method terminated in",stoptime-starttime,"seconds.")
print("Final energy:",energy)
print("Final epsilon:",epsilon)
print("Ground state covariance matrix:")
print(Omega)

plt.plot(range(len(energies)),energies)
plt.show()
