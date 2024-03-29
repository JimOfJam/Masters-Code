import numpy as np
import gaussian as g
import scipy.linalg
import scipy.sparse 
import matplotlib.pyplot as plt 
import timeit 

i = 0+1j
n = 200

blank = "                                                                                                    "

# Helps us index the Majorana modes correctly
def index(j,sigma,k):
    return int(j % (n/2) + sigma*(n/2) + k*n)

# The h_{ab} matrix which contracts with the Majorana modes to produce the Hamiltonian
h = np.zeros((2*n,2*n))
rows = []
columns = []
data = []
for j in range(int(n/2)):
    for sigma in range(2):
        rows += [index(j,sigma,0),index(j,sigma,1),index(j+1,sigma,0),index(j+1,sigma,1)]
        columns += [index(j+1,sigma,1),index(j+1,sigma,0),index(j,sigma,1),index(j,sigma,0)]
        data += [-1,1,-1,1]
h = scipy.sparse.csc_array((data,(rows,columns)))
h = 0.5*(h - h.transpose())


# Define the basis elements of the Lie algebra 
basisK = []
lieDim = n*(2*n-1)
count = 0
for j in range(2*n):
    for k in range(j+1,2*n):
        rows = [j,k]
        columns = [k,j]
        data = [1,-1]
        basisK.append(scipy.sparse.csc_array((data,(rows,columns)),shape=(2*n,2*n)))

        count += 1
        if count % print("Setting up the basis of the Lie algebra...")

for v in basisK:
    v = v/(v@(v.transpose())).trace()

def RtoT(vector):
    index = 0
    K = np.zeros((2*n,2*n))
    for j in range(2*n):
        for k in range(j+1,2*n):
            K[j,k] = vector[index]
            K[k,j] = -vector[index]
            index += 1
    return K 

def lieProduct(A,B):
    return np.trace(np.matmul(A,B.T))



print(blank,end="\r")


# This is the function to be optimised 
def expectationValue(x):
    return -0.25*np.trace(h@x)

def gradient(x):
    grad = np.zeros(lieDim)
    for j in range(lieDim):
        grad[j] = expectationValue(basisK[j]@x+x@basisK[j].T)
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

maxIterations = 50
starttime = timeit.default_timer()
for j in range(maxIterations):
    grad = gradient(Omega)
    candidate = vary(Omega,RtoT(grad),-epsilon)
    candidateEnergy = expectationValue(candidate).real

    if candidateEnergy < energy:
        Omega = candidate
        energy = candidateEnergy 
    elif epsilon < 10**(-5):
        print("Epsilon reached threshold after",j+1,"iterations.")
        break 
    else:
        epsilon = epsilon*0.5
    energies.append(energy)

    print("Progress: " + str(j/maxIterations*100) + "%, Current epsilon: " + str(epsilon) + ", Current energy: " + str(energy),end="\r") 
    energies.append(energy)

print(blank,end="\r")
stoptime = timeit.default_timer()

print("Method terminated in",stoptime-starttime,"seconds.")
print("Final energy:",energy)
print("Final epsilon:",epsilon)
#print("Ground state covariance matrix:")
#print(Omega)

plt.plot(range(len(energies)),energies)
plt.show()
