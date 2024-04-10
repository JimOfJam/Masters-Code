import numpy as np
import gaussian as g
import scipy.linalg
import scipy.sparse 
import matplotlib.pyplot as plt 
import timeit 
from sympy.combinatorics import Permutation, PermutationGroup

i = 0+1j


n = 4
t = 1
U = 10


blank = "                                                                                                    "

# Helps us index the Majorana modes correctly.
def index(j,sigma,k):
    if sigma == "up":
        sigma = 1
    elif sigma == "down":
        sigma = 0
    if k == "q":
        k = 0
    elif k == "p":
        k = 1
    return int(j % (n/2) + sigma*(n/2) + k*n)

# Converts a 4-tensor into an alternating 4-tensor.
def Alt(tensor):
    permutations = PermutationGroup(Permutation(0,1),Permutation(0,1,2,3))
    result = []
    for i in tensor:
        for perm in permutations:
            result += [perm([i[0],i[1],i[2],i[3]]) + [-(-1)**perm.parity()/24*i[4]]]
    return result
    

# The h1_{ab} matrix which contracts with the Majorana modes to produce the quadratic part of the Hamiltonian
rows = []
columns = []
data = []
for j in range(int(n/2)):
    for sigma in range(2):
        rows += [index(j,sigma,0),index(j,sigma,1),index(j+1,sigma,0),index(j+1,sigma,1)]
        columns += [index(j+1,sigma,1),index(j+1,sigma,0),index(j,sigma,1),index(j,sigma,0)]
        data += [-t,t,-t,t]
h1 = scipy.sparse.csc_array((data,(rows,columns)),shape=(2*n,2*n))

# The h2_{abcd} tensor that produces the quartic part of the Hamiltonian
sparseh2 = []
for k in range(int(n/2)):
    sparseh2 += [
        [index(k,'up','q'),index(k,'up','q'),index(k,'down','q'),index(k,'down','q'),1.0],
        [index(k,'up','q'),index(k,'up','q'),index(k,'down','q'),index(k,'down','p'),1.0j],
        [index(k,'up','q'),index(k,'up','q'),index(k,'down','p'),index(k,'down','q'),-1.0j],
        [index(k,'up','q'),index(k,'up','q'),index(k,'down','p'),index(k,'down','p'),1.0],
        [index(k,'up','q'),index(k,'up','p'),index(k,'down','q'),index(k,'down','q'),1.0j],
        [index(k,'up','q'),index(k,'up','p'),index(k,'down','q'),index(k,'down','p'),-1.0],
        [index(k,'up','q'),index(k,'up','p'),index(k,'down','p'),index(k,'down','q'),1.0],
        [index(k,'up','q'),index(k,'up','p'),index(k,'down','p'),index(k,'down','p'),1.0j],
        [index(k,'up','p'),index(k,'up','q'),index(k,'down','q'),index(k,'down','q'),-1.0j],
        [index(k,'up','p'),index(k,'up','q'),index(k,'down','q'),index(k,'down','p'),1.0],
        [index(k,'up','p'),index(k,'up','q'),index(k,'down','p'),index(k,'down','q'),-1.0],
        [index(k,'up','p'),index(k,'up','q'),index(k,'down','p'),index(k,'down','p'),-1.0j],
        [index(k,'up','p'),index(k,'up','p'),index(k,'down','q'),index(k,'down','q'),1.0],
        [index(k,'up','p'),index(k,'up','p'),index(k,'down','q'),index(k,'down','p'),1.0j],
        [index(k,'up','p'),index(k,'up','p'),index(k,'down','p'),index(k,'down','q'),-1.0j],
        [index(k,'up','p'),index(k,'up','p'),index(k,'down','p'),index(k,'down','p'),1.0]
    ]


# Define the basis elements of the Lie algebra 
basisK = []
lieDim = n*(2*n-1)
count = 0
progress = 0.01
for j in range(2*n):
    for k in range(j+1,2*n):
        rows = [j,k]
        columns = [k,j]
        data = [1,-1]
        basisK.append(scipy.sparse.csc_array((data,(rows,columns)),shape=(2*n,2*n)))

        if j * k / (2 * lieDim) > progress:
            progress += 0.01
            print("Constructing basis of Lie algebra... Progress: " + str(round(j * k / (2 * lieDim) * 100, 2)) + "%.", end="\r")
for v in basisK:
    v = v/(v @ (v.transpose())).trace()

# Takes a list of coefficients as input. Outputs the Lie algebra element corresponding to those coefficients. 
def RtoT(vector):
    index = 0
    K = np.zeros((2*n,2*n),dtype="complex")
    for j in range(2*n):
        for k in range(j+1,2*n):
            K[j,k] = vector[index]
            K[k,j] = -vector[index]
            index += 1
    return K 


print(blank,end="\r")


# This is the function to be optimised 
def expectationValue(x):
    C = 0.5*(np.eye(2*n)+1.0j*x)
    # Constant shift 
    value = 0
    # Quadratic contribution
    value += 0.25*np.trace(h1 @ x)
    # Quartic contribution
    newpart = 0
    for row in sparseh2:
        a = row[0]; b = row[1]; c = row[2]; d = row[3]; h = row[4]
        newpart += 0.25*U*h*(C[a][b]*C[c][d] - C[a][c]*C[b][d] + C[a][d]*C[b][c])
    value += newpart 
    return value.real

def gradient(x):
    grad = np.zeros(lieDim,dtype="complex_")
    C = np.eye(2*n,dtype="complex_") + 1j*x
    for j in range(lieDim):
        variance = basisK[j] @ x + x @ basisK[j].T
        grad[j] = 0.25*np.trace(h1 @ variance)
        for row in sparseh2:
            a = row[0]; b = row[1]; c = row[2]; d = row[3]; h = row[4]
            grad[j] += 1/4*h*(variance[a][b]*C[c][d] + C[a][b]*variance[c][d] - variance[a][c]*C[b][d] - C[a][c]*variance[b][d] + variance[a][d]*C[b][c] + C[a][d]*variance[b][c])
    return grad

def vary(x,K,epsilon):
    return g.prodN([scipy.linalg.expm(epsilon*K),x,scipy.linalg.expm(epsilon*(K.T))])

# Omega is the standard symplectic form
Omega = np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
# K is a random Lie algebra element
K = RtoT(np.random.rand(lieDim))
#Omega is the corresponding random group element
Omega = vary(Omega,K,100)

energies = []
energy = expectationValue(Omega)
print("Initial energy: ",energy)
epsilon = 10
maxIterations = 20

for j in range(maxIterations):
    grad = gradient(Omega).real
    candidate = vary(Omega,-RtoT(grad),epsilon)
    candidateEnergy = expectationValue(candidate)
    print(candidateEnergy)

    if candidateEnergy < energy:
        Omega = candidate
        energy = candidateEnergy
    else:
        epsilon = epsilon*0.5
    
    energies.append(energy)

print("final energy:",energy)

plt.plot(range(maxIterations),energies)
plt.show()

