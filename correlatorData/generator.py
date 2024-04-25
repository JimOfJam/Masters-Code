import numpy as np
import scipy.sparse as sparse
import timeit

# Wick's theorem for 4-point functions
def wicks4(C,a,b,c,d):
    return C[a,b]*C[c,d] - C[a,c]*C[b,d] + C[a,d]*C[b,c]

def generateCorrelators(N):

    start = timeit.default_timer()

    n = 2*N

    Omega0 = np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
    Omega0 = sparse.csc_matrix(Omega0)
    x = range(2*n)

    C = sparse.csc_matrix(np.eye(2*n) + 1j*Omega0)

    vacuumCorrelator = []
    for i in x:
        for j in x:
            for k in x:
                for l in x:
                    wicks = wicks4(C,i,j,k,l)
                    if not wicks == 0:
                        vacuumCorrelator.append([i,j,k,l,wicks])
    
    filename = "correlatorData/n=" + str(n) + ".txt"
    with open(filename,'w') as f:
        f.write("vacuumCorrelator = " + str(vacuumCorrelator))

    stop = timeit.default_timer()
    print("Time taken for n =",n,":",stop-start,"seconds.")

generateCorrelators(10)
