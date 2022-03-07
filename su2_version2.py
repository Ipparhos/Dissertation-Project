import numpy as np
np.set_printoptions(precision=16)
from random_su2 import*
from numpy.linalg import multi_dot
from unitarity_check import*
import copy

N = 7                                                           # N is the number of lattice points for each dimension
D = 4                                                           #Number of mi, ni arguments
group_dim = 2                                                   # group_dim is the dimensions of the lattice
U = np.zeros ((group_dim,group_dim,2*D+1,N**D+1), np.complex128)    #4D lattice link variables
nn = np.zeros((2*D+1,N**D+1),np.int64)                             #Nearest neghbors
N_configurations = 10000                                        #Times of configurations(sweeps) we want to occure
beta = 6.0                                                      #Coupling constant β
eps = 0.24                                                      #Random parameter that controls the acceptance ratio
loop_list = []                                                  #Here we save the Wilson Loop measurements
static_potential_list = []                                      #Here we save the Static Potential measurements
plaquete_size = N                                               #The size of the plaquete we want to loop around
N_thermalazation = 500                                          #Number of configurations until thermalization
N_Metropolis = 1                                                #Number of Metropolis iterations per configuration
N_Overrelaxation = 0                                            #Number of Overrelaxation iterations per configuration
N_Heatbath = 0                                                  #Number of Heatbath iterations per configuration

def nearest_neghbors():
    x = np.zeros(D+1,np.int64)
    xnn = np.zeros(D+1,np.int64)
    for n in range(1,N**D+1):
        LHS = n-1
        x[D] = LHS/(N**(D-1))+1

        for j in range (D-1,0,-1):
            LHS  = LHS - (x[j+1]-1) * N**(j)
            x[j] = LHS / (N**(j-1)) + 1
        
        for mi in range (1,D):
            xnn = copy.copy(x)
            xnn [mi] = x[mi] + 1
            if (xnn[mi] > N):
                xnn[mi] = xnn[mi] - N
            nn [mi,n] = 0

            for j in range(1,D+1):
                nn[mi,n] = nn[mi,n]  + (xnn[j]-1) * N**(j-1)
            nn[mi,n] = nn[mi,n] + 1

            xnn[mi] = x[mi] - 1
            if(xnn[mi] < 1 ):
                xnn[mi] = xnn[mi] + N
            nn [-(mi),n] = 0

            for j in range(1,D+1):
                nn[-(mi),n] = nn[-(mi),n] + (xnn[j]-1) * N**(j-1)
            nn [-(mi),n] = nn [-(mi),n] + 1

        #Temporal dimension

        mi = D
        xnn = copy.copy(x)
        xnn[mi]  = x[mi] + 1
        if(xnn[mi] > N):
            xnn[mi] = xnn[mi] - N
        nn[mi,n]  = 0

        for j in range(1,D+1):
            nn[mi,n]  = nn[mi,n] + (xnn[j]-1) * N**(j-1)
        nn[mi,n]  = nn[mi,n] + 1

        xnn[mi]  = x[mi] - 1
        if(xnn[mi] < 1 ):
            xnn[mi] = xnn[mi] + N
        nn [-(mi),n]  = 0

        for j in range(1,D+1):
            nn[-mi,n]  = nn[-mi,n] + (xnn[j]-1) * N**(j-1)
        nn[-mi,n] = nn[-mi,n] + 1
        print(n,'||',nn[:,n])

def initialize_lattice(parameter=0):
    for n in range(0,N**D+1):
        for mi in range(0,D):

            if  parameter == 0 :
                U[:,:,mi,n] = np.identity(group_dim)                            
            else:
                U[:,:,mi,n] = generate_su2_U(eps)

            
def calculate_staple(n,mi):
    staple = np.zeros((group_dim, group_dim) , np.complex128)
    
    for ni in range(-D,D + 1):
        if ( ni ==0 or ni == mi or ni == -(mi)):
            continue
        staple += Amn(mi,ni,n)
    
    return staple

def Amn(mi,ni,n):
    n2 = nn[mi,n]
    n3 = nn[ni,n2]
    n4 = nn[ni,n] 


    U2 = U[:,:, ni,n2]
    U3 = U[:,:,-mi,n3]
    U4 = U[:,:,-ni,n4]

    V = U2
    V = np.matmul(V,U3)
    V = np.matmul(V,U4)
    return V

def calculate_S(n,mi,staple):
    trace_of_UA = (np.trace(np.matmul( U[:,:,mi,n], staple))).real
    S = -(beta*trace_of_UA) / group_dim
    return S

def Metropolis(n,mi):
    accept = False

    staple = calculate_staple(n,mi)
    old_link = U[:,:,mi,n].copy()
    old_S = calculate_S(n,mi,staple)    
    U[:,:,mi,n] = np.matmul(generate_su2_U(eps) , old_link)
    new_S = calculate_S(n,mi,staple)
    dS = new_S - old_S
 
    #Acceptance Condition
    if ( (dS > 0.0) and (np.exp(-dS) < np.random.uniform(0,1)) ):
        U[:,:,mi,n] = old_link
        accept = True

    if accept == True:
        U[:,:,-mi,nn[mi,n]] = U[:,:,mi,n].conj().T
    #print(U[:,:,mi,n])
    #Checking for unitarity violation and re-unitarise if necessary
    # U[:,:,mi,n] = check_unitarity(U[:,:,mi,n])

def update_lattice():

    for n in range(1,N**D+1):
        for mi in range(1,D+1):
            for i in range(N_Metropolis):
                Metropolis(n,mi)

def measure_static_potential():
    pmpn = np.zeros (N+2, np.complex128)
    count = np.zeros (N+2, np.int64)
    pm = np.zeros(N**3+1,np.complex128)
    
    for n in range(1,N**3+1):
        p = calculate_polyakov_line(n)
        pm[n] = np.trace(p)

    for n in range(1,N**3+1):

        for mi in range(1,D):
            m = n 
            for r in range(1,N+2):
                pn = pm[m]
                pmpn[r] = pmpn[r] + pm[n]*pn.conj()
                count[r] = count[r] + 1
                m = nn[mi,m]
                
    return pmpn/count

def calculate_polyakov_line(n):

    mi = D-1
    P = U[:,:,mi,n]
    m = n
    for n in range(0,N-1):
        m = nn[mi,m]
        P = np.matmul(P,U[:,:,mi,n])
    return P

def MCaverage():
    sp = []
    
    
    for i in range(N_configurations + N_thermalazation):
        update_lattice()
        
        #Start measuring after the N_thermalazation configurations
        if (i > N_thermalazation-1):
             
            sp.append(measure_static_potential() )
             
            
        #Printing the results
        #Writing Static Potential
    f = open(f"SP2_M{N_Metropolis}O{N_Overrelaxation}H{N_Heatbath}_N{N}_S{N_configurations}_b{beta}", "w") 
    for i in range(N_configurations):
        f.write(f"{(sp[i][1]).real}\t{(sp[i][2]).real}\t{(sp[i][3]).real}\t{(sp[i][4]).real}\t{(sp[i][5]).real}\t{(sp[i][6]).real}\t{(sp[i][7]).real}\t{(sp[i][8]).real}\n")
    f.close()        


#-----------------------------------Running the Program----------------------------------------------------------


nearest_neghbors()
initialize_lattice()                                        #Initializing the lattice. Choose 0 or leave blank for cold 
                                                            #Choose 1 for hot start
MCaverage()

