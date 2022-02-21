from pickle import FALSE
from unittest import skip
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=16)
from random_su2 import*
from numpy.linalg import multi_dot
from unitarity_check import*
from overrelaxation_debugging import*


N = 7                                                           # N is the number of lattice points for each dimension
N_mi = 4                                                        #Number of mi, ni arguments
group_dim = 2                                                   # group_dim is the dimensions of the lattice
U = np.zeros ((N,N,N,N,N_mi,group_dim,group_dim), np.complex128)    #4D lattice link variables
N_configurations = 10000                                            #Times of configurations(sweeps) we want to occure
beta = 2.0                                                      #Coupling constant β
eps = 0.24                                                      #Random parameter that controls the acceptance ratio
loop_list = []                                                  #Here we save the Wilson Loop measurements
static_potential_list = []                                      #Here we save the Static Potential measurements
plaquete_size = N                                               #The size of the plaquete we want to loop around
N_thermalazation = 500                                          #Number of configurations until thermalization
N_Metropolis = 0                                                #Number of Metropolis iterations per configuration
N_Overrelaxation = 1                                            #Number of Overrelaxation iterations per configuration
N_Heatbath = 1                                                  #Number of Heatbath iterations per configuration



#Here we initialize the lattice with two options. 0 for cold start and 1 for hot
def initialize_lattice(parameter=0): #parameter is 0 for cold start and 1 for hot start
    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mi in range(N_mi):   # mi is the directional index
                        if  parameter == 0 :
                            U[t,x,y,z,mi] = np.identity(group_dim)                            
                        else:
                            U[t,x,y,z,mi] = generate_su2_U(eps)

def calculate_staple(t,x,y,z,mi):
    staple = np.zeros((group_dim, group_dim) , np.complex128)
    mi_hat = [0,0,0,0]
    mi_hat[mi] = 1
    for ni in range(N_mi):
        if ni != mi:
            ni_hat = [0,0,0,0]
            ni_hat[ni] = 1
            staple += multi_dot([
                U[(t+mi_hat[0])%N , (x+mi_hat[1])%N , (y+mi_hat[2])%N , (z+mi_hat[3])%N , ni],
                U[(t+ni_hat[0])%N , (x+ni_hat[1])%N , (y+ni_hat[2])%N , (z+ni_hat[3])%N , mi].conj().T,
                U[t , x , y , z , ni].conj().T])

            staple += multi_dot([
                U[(t+mi_hat[0]-ni_hat[0])%N , (x+mi_hat[1]-ni_hat[1])%N , (y+mi_hat[2]-ni_hat[2])%N , (z+mi_hat[3]-ni_hat[3])%N , ni].conj().T,
                U[(t-ni_hat[0])%N , (x-ni_hat[1])%N , (y-ni_hat[2])%N , (z-ni_hat[3])%N , mi].conj().T,
                U[(t-ni_hat[0])%N , (x-ni_hat[1])%N , (y-ni_hat[2])%N , (z-ni_hat[3])%N , ni]])
    return staple

def calculate_S(t,x,y,z,mi,staple):
    trace_of_UA = (np.trace(np.dot( U[t,x,y,z,mi], staple))).real
    S = -(beta*trace_of_UA) / group_dim
    return S


def Overrelaxation(t,x,y,z,mi):
    
    staple = calculate_staple(t,x,y,z,mi)
    old_link = U[t,x,y,z,mi].copy()
    old_S = calculate_S(t,x,y,z,mi,staple)
           
        
    Uo = staple.conj().T / np.sqrt(np.linalg.det(staple))
    U[t,x,y,z,mi] = multi_dot([Uo,old_link.conj().T,Uo]) 
        
    new_S = calculate_S(t,x,y,z,mi,staple)
    dS = new_S - old_S
    # ov_debug(Uo,dS)
    
    
    #Acceptance Condition
    if ( (dS > 0.0) and (np.exp(-dS) < np.random.uniform(0,1)) ):
        U[t,x,y,z,mi] = old_link
        
    #Checking for unitarity violation and re-unitarise if necessary
    U[t,x,y,z,mi] = check_unitarity(U[t,x,y,z,mi])


def Metropolis(t,x,y,z,mi):
    
    staple = calculate_staple(t,x,y,z,mi)
    old_link = U[t,x,y,z,mi].copy()
    old_S = calculate_S(t,x,y,z,mi,staple)    
        
    U[t,x,y,z,mi] = np.dot(generate_su2_U(eps) , old_link)
 
    new_S = calculate_S(t,x,y,z,mi,staple)
    dS = new_S - old_S
 
    #Acceptance Condition
    if ( (dS > 0.0) and (np.exp(-dS) < np.random.uniform(0,1)) ):
        U[t,x,y,z,mi] = old_link

    
        
    #Checking for unitarity violation and re-unitarise if necessary
    U[t,x,y,z,mi] = check_unitarity(U[t,x,y,z,mi])

def Heatbath(t,x,y,z,mi):          
    A = calculate_staple(t,x,y,z,mi)
    a = np.sqrt(np.linalg.det(A)).real
    if(a == 0.):
        r0  = np.random.uniform(-0.5,0.5)
        x0  = np.sign(r0)*np.sqrt(1-eps**2)
        
        r   = np.random.random((3)) - 0.5      
        x   = eps*r/np.linalg.norm(r)

        U[t,x,y,z,mi] = x0*np.identity(2) + 1j*x[0]*sx + 1j*x[1]*sy + 1j*x[2]*sz
    else:

        #Generating x0
        random_generated_number = np.random.random(3)        
        lambda2 = (np.log(random_generated_number[0]) + (np.cos(2*np.pi*random_generated_number[1])**2)*np.log(random_generated_number[2]))/(-2*a*beta)    
        while(np.random.random()**2 > 1 - lambda2):
            random_generated_number = np.random.random(3)
            lambda2 = (np.log(random_generated_number[0]) + (np.cos(2*np.pi*random_generated_number[1])**2)*np.log(random_generated_number[2]))/(-2*a*beta)
        x0 = 1- 2*lambda2

        #Generating x1 x2 x3
        x_vector = 2*np.random.random(3) - 1
        while( x_vector[0]**2 + x_vector[1]**2 + x_vector[2]**2 > 1):
            x_vector = 2*np.random.random(3) - 1        
        x_vector *= np.sqrt(1-x0**2)/np.sqrt(x_vector[0]**2 + x_vector[1]**2 + x_vector[2]**2)

        #Building the X matrix of SU(2)
        X = x0*np.identity(2) + 1j*x_vector[0]*sx + 1j*x_vector[1]*sy + 1j*x_vector[2]*sz        

        U[t,x,y,z,mi] = np.dot(X , A.conj().T/a)

    #Checking for unitarity violation and re-unitarise if necessary
    U[t,x,y,z,mi] = check_unitarity(U[t,x,y,z,mi])


def update_lattice():

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mi in range(N_mi):
                        for i in range(N_Metropolis):
                            Metropolis(t,x,y,z,mi)
                        for i in range(N_Heatbath):
                            Heatbath(t,x,y,z,mi)
                        for i in range(N_Overrelaxation):  
                            Overrelaxation(t,x,y,z,mi)
                        
                            
                                

def measure_polyakov_loop():
    polyakov_trace = 0 
    for x in range(N):
            for y in range(N):
                for z in range(N): 
                    polyakov_loop = np.identity(group_dim)
                    for t in range(N):
                        polyakov_loop = np.dot(polyakov_loop,U[t,x,y,z,0])
                    polyakov_trace += np.trace(polyakov_loop)
    return polyakov_trace/(group_dim*N**3)

def measure_static_potential(): # runs for each N_configuration
    pmpn = np.zeros (N+1, np.complex128)
    count = np.zeros (N+1, np.int64)
    pm = np.zeros((N,N,N),np.complex128)
    for x in range(N):
        for y in range(N):
            for z in range(N):
                pm[x,y,z] = calculate_polyakov_line(x,y,z)

    for x in range(N):
        for y in range(N):
            for z in range(N):

                for mi in range(1,4):
                    mi_hat = [0,0,0,0]
                    mi_hat[mi] = 1
                    for r in range(0,N+1):
                        pn = pm[(x + mi_hat[1]*r)%N, (y + mi_hat[2]*r)%N , (z + mi_hat[3]*r)%N]
                        pmpn[r] = pmpn[r] + pm[x,y,z]*(pn.conj())
                        count[r] = count[r] + 1
    return pmpn/count

def calculate_polyakov_line(x,y,z):

    P = U[0,x,y,z,0]
    for t in range(1,N):
        P = np.dot(P,U[t,x,y,z,0])
    return np.trace(P)

def Average_Wilson_plaquette(i,j):
    sum_averege_plaqutte = 0.0
    plaquette_av = []
    
    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mi in range(N_mi-1):
                        mi_hat = [0,0,0,0]
                        mi_hat[mi] = 1
                        for ni in range(mi+1,N_mi): 
                            ni_hat = [0,0,0,0]
                            ni_hat[ni] = 1
                            I=0
                            J=0
                            
                            temp = np.identity(group_dim)
                            
                            #Horisontal displacement to the left
                            for I in range(0,i):
                                temp = np.dot(temp, U[(t+I*mi_hat[0])%N , (x+I*mi_hat[1])%N , (y+I*mi_hat[2])%N , (z+I*mi_hat[3])%N ,mi])
                            #Vertical displacement upwards
                            for J in range(0,j):
                                temp = np.dot(temp, U[(t+ (I+1)*mi_hat[0] + J*ni_hat[0])%N , (x+ (I+1)*mi_hat[1]+ J*ni_hat[1])%N , (y+ (I+1)*mi_hat[2]+ J*ni_hat[2])%N , (z+ (I+1)*mi_hat[3]+ J*ni_hat[3])%N , ni])
                            #Horisontal displacement to the right
                            for I in range(I,-1,-1):
                                temp = np.dot(temp, U[(t+ I*mi_hat[0] + (J+1)*ni_hat[0])%N , (x+ I*mi_hat[1]+ (J+1)*ni_hat[1])%N , (y+ I*mi_hat[2]+ (J+1)*ni_hat[2])%N , (z+ I*mi_hat[3]+ (J+1)*ni_hat[3])%N , mi].conj().T)
                            #Vertical displacement downwards
                            for J in range(J,-1,-1):
                                temp = np.dot(temp, U[(t+ J*ni_hat[0])%N , (x+ (J)*ni_hat[1])%N , (y+ (J)*ni_hat[2])%N , (z+ (J)*ni_hat[3])%N , ni].conj().T )
                            plaquette_av.append((np.trace(temp)).real)                               
                            sum_averege_plaqutte += (np.trace(temp)).real
    return (sum_averege_plaqutte)/(6*N**4),plaquette_av

def Wilson_loop_static_potential():
    W = np.zeros (N+1, np.complex128)
    count = np.zeros (N+1, np.int64)
    for x in range(N):
        for y in range(N):
            for z in range(N):
                
                for mi in range(1,4):
                    mi_hat = [0,0,0,0]
                    mi_hat[mi] = 1
                    for r in range(N+1):
                        S_mn = calculate_spatial_transporter(0,x,y,z,(x + mi_hat[1]*r)%N, (y + mi_hat[2]*r)%N , (z + mi_hat[3]*r)%N)
                        T_n = calculate_temporal_transporter((x + mi_hat[1]*r)%N, (y + mi_hat[2]*r)%N , (z + mi_hat[3]*r)%N)
                        S_nm = calculate_spatial_transporter(N,x,y,z,(x + mi_hat[1]*r)%N, (y + mi_hat[2]*r)%N , (z + mi_hat[3]*r)%N)
                        T_m = calculate_temporal_transporter(x,y,z)

                        W[r] += np.trace(multi_dot([S_mn, T_n.conj().T, S_nm.conj().T,T_m]))
                        count[r] += 1
    return np.divide(W,count)


def calculate_temporal_transporter(x,y,z):

    Temporal = np.identity(group_dim)
    for t in range(N):
        Temporal = np.dot(Temporal,U[t,x,y,z,0])
    return Temporal

def calculate_spatial_transporter(t,xm,ym,zm,xn,yn,zn):
    
    S = np.identity(group_dim)
    for x in range(xm,xn):
        S = np.dot(S,U[t,x,ym,zm,1])
    for y in range(ym,yn):
        S = np.dot(S,U[t,xn,y,zm,2])
    for z in range(zm,zn):
        S = np.dot(S,U[t,xn,yn,z,3])
    return S



#Prints the average of Monte_Carlo_loop method
def MCaverage():
    sp = []
    sp_wilson = 0
    s = 0
    p = 0
    sum = []
    p_av = []
    for i in range(N_configurations + N_thermalazation):
        update_lattice()
        
        #Start measuring after the N_thermalazation configurations
        if (i > N_thermalazation-1):
            # polyakov_loop = measure_polyakov_loop() 
            sp.append(measure_static_potential())
            #sp_wilson = Wilson_loop_static_potential()
            s , p = Average_Wilson_plaquette(1,1)
            sum.append(s)
            p_av.append(p)

    #Printing the results
    f = open(f"SP_M{N_Metropolis}O{N_Overrelaxation}H{N_Heatbath}_N{N}_S{N_configurations}_b{beta}", "w")
    g = open(f"AP_M{N_Metropolis}O{N_Overrelaxation}H{N_Heatbath}_N{N}_S{N_configurations}_b{beta}", "w")             
    for i in range (N_configurations):       
    #Writing Static Potential            
        
        f.write(f"{(sp[i][0]).real}\t{(sp[i][1]).real}\t{(sp[i][2]).real}\t{(sp[i][3]).real}\t{(sp[i][4]).real}\t{(sp[i][5]).real}\t{(sp[i][6]).real}\t{(sp[i][7]).real}\n")
        
    #Writing Static Potential            
        #f = open(f"SPW_M{N_Metropolis}O{N_Overrelaxation}H{N_Heatbath}_N{N}_S{N_configurations}_b{beta}", "a")
        #f.write(f"{(sp_wilson[0]).real}\t{(sp_wilson[1]).real}\t{(sp_wilson[2]).real}\t{(sp_wilson[3]).real}\n")
        #f.close()

    #Writing Average Plaquette
        
        g.write(f"{(sum[i]).real}\t{(p_av[i][0]).real}\t{(p_av[i][1]).real}\t{(p_av[i][2]).real}\t{(p_av[i][3]).real}\t{(p_av[i][4]).real}\t{(p_av[i][5]).real}\t{(p_av[i][6]).real}\n")

    f.close()
    g.close()

        
    



#-----------------------------------Running the Program----------------------------------------------------------



initialize_lattice()                                        #Initializing the lattice. Choose 0 or leave blank for cold 
                                                            #Choose 1 for hot start
MCaverage()