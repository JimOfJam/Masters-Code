import numpy as np 
import matplotlib.pyplot as plt
import timeit
import scipy.linalg as sp

np.set_printoptions(precision=4,suppress=True)
j = 0+1j

paulix = np.array([[0,1],[1,0]],dtype='complex_')
pauliy = np.array([[0,-j],[j,0]],dtype='complex_')
pauliz = np.array([[1,0],[0,-1]],dtype='complex_')
paulip = np.array([[0,1],[0,0]],dtype='complex_')
paulim = np.array([[0,0],[1,0]],dtype='complex_')

def innerProd(a,b):
    return np.matmul(adjoint(a),b)[0][0]

def kronN(list):
    result = list[0]
    for i in range(1,len(list)):
        result = np.kron(result,list[i])
    return result

def prodN(list):
    result = list[0]
    for i in range(1,len(list)):
        result = np.matmul(result,list[i])
    return result 

def randomC():
    return np.random.rand() + np.random.rand()*j

def adjoint(array):
    return np.conjugate(array.T)


class FermionicSpace:
    def __init__(self,n):
        print("Initialising fermionic Hilbert space of dimension %s..." % (2**n,))
        starttime = timeit.default_timer()
        if n==1:
            self.raising = [paulip]
            self.lowering = [paulim]
        else:
            self.raising = []
            for i in range(n):
                factors = []
                for k in range(n):
                    if k<i:
                        factors.append(-pauliz)
                    elif k>i:
                        factors.append(np.eye(2))
                    else:
                        factors.append(paulim)
                self.raising.append(kronN(factors)) # Taking a kronecker product in this way has computational complexity O(2^(4n)). It can be done in polynomial time by observing a pattern in the matrices and populating the matrix with the necessary values directly. 

            self.lowering = []
            for i in range(n):
                factors = []
                for k in range(n):
                    if k<i:
                        factors.append(-pauliz)
                    elif k>i:
                        factors.append(np.eye(2))
                    else:
                        factors.append(paulip)
                self.lowering.append(kronN(factors))

        
                
        self.p = []
        self.q = []
        for i in range(n):
            self.q.append((self.lowering[i] + self.raising[i])/(np.sqrt(2)))
            self.p.append((self.lowering[i] - self.raising[i])/(j*np.sqrt(2)))
        self.majorana = self.q+self.p
        
        self.J0 = np.zeros((2*n,2*n),dtype="complex_")
        for i in range(2*n):
            if i%2==0:
                self.J0[i][i+1]=1
            else:
                self.J0[i][i-1]=-1

        self.h0 = self.J0
        
        self.J1 = np.block([[np.zeros((n,n),dtype="complex_"),np.eye(n)],[-np.eye(n),np.zeros((n,n),dtype="complex_")]])
        
        self.hamiltonian = None
        self.groundState = None

        self.states = []
        self.dof = n
        self.dimension = 2**n

        stoptime = timeit.default_timer()
        print("Hilbert space initialised with runtime %s seconds." % (stoptime-starttime,))
    
    def randomState(self):
        vector = []
        for i in range(self.dimension):
            vector.append(randomC())
        return adjoint(np.array(vector))/np.linalg.norm(vector)
    
    def randomOrthogonal(self,psi):
        phi = self.randomState()
        phiOrth = phi - np.matmul(adjoint(psi),phi)*psi
        return phiOrth/np.linalg.norm(phiOrth)  

    def addState(self,vector = None):
        if vector is None:
            vector = self.randomState()
        vector = np.array(vector,dtype='complex_')[None]
        self.states.append(adjoint(vector)/np.linalg.norm(vector))

    def correlator(self,a,b,state):
        z = []
        for i in range(2*self.dof):
            z.append(prodN([adjoint(state),self.majorana[i],state])[0][0])
        
        c = prodN([adjoint(state),self.majorana[a]-z[a]*np.eye(self.dimension),self.majorana[b]-z[b]*np.eye(self.dimension),state])[0][0]
        return c
    
    def correlatorMatrix(self,state):
        z = []
        for i in range(2*self.dof):
            z.append(prodN([adjoint(state),self.majorana[i],state])[0][0])
        
        c = np.zeros((2*self.dof,2*self.dof),dtype="complex_")
        for i in range(2*self.dof):
            for k in range(2*self.dof):
                c[i][k] = prodN([adjoint(state),self.majorana[i]-z[i]*np.eye(self.dimension),self.majorana[k]-z[k]*np.eye(self.dimension),state])[0][0]
        return c
    
    def addGaussian(self,lie = None):
        if lie is None:
            lie = np.zeros((2*self.dof,2*self.dof),dtype="complex_")
            for i in range(2*self.dof):
                for k in range(2*self.dof):
                    lie[i][k]=randomC()
            lie = lie - lie.T

        hamiltonianLie = np.zeros((self.dimension,self.dimension),dtype="complex_")
        for i in range(2*self.dof):
            for k in range(2*self.dof):
                hamiltonianLie = hamiltonianLie + j/2*lie[i][k]*prodN([self.majorana[i],self.majorana[k]])
        
        gaussianUnitary = sp.expm(-j*hamiltonianLie)

        self.states.append(np.matmul(gaussianUnitary,self.groundState))

    def covariance(self,state):
        C = self.correlatorMatrix(state)
        Omega = j*(C.T - C)
        return Omega
    
    def isGaussian(self,state,tolerance=10**(-14)):
        C = self.correlator(state)
        G = C + C.T 
        Omega = j*(C.T - C)
        g = np.linalg.inv(G)
        J = np.matmul(Omega,g)
        
        if np.linalg.norm(np.matmul(J,J)+np.eye(2*self.dof)) < tolerance:
            return True 
        else:
            return False
        
    def deleteStates(self,list):
        for i in list:
            self.states = self.states[:i] + self.states[i+1:]


    def genWicks(self,a,b,state):        
        return prodN([adjoint(state),self.majorana[a],self.majorana[b],self.groundState])[0][0]
    

class fermionicPhaseSpace:
    def __init__(self,metric,storageType="covariance"):
        self.storageType = storageType
        self.metric = metric 
        self.gaussians = []
        self.metricInv = np.linalg.inv(metric)
    
    def addGaussian(self,covariance):
        self.gaussians.append(covariance)

    def complexStructure(self,covariance):
        return np.matmul(covariance,self.metricInv)
    
    def covariance(self,complexStructure):
        return np.matmul(complexStructure,self.metric)
    
