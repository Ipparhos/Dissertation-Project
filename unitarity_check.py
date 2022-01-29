import numpy as np
np.set_printoptions(precision=16)

def check_unitarity(U):
    epsilon = 1.*10**(-15) #Limit of re-unitarization    
        
    result = np.abs(np.linalg.det(U)-1)

    if result > epsilon : 
        
        U = reunitarise(U)
        result = np.abs(np.linalg.det(U)-1)
        
        if result > epsilon: 
            
            print("Failed to Reunitarise!")
            #print(result)
            
        else:
            pass
            #print(result)
            #print("------------------------------")
    else:
        pass
        #print("Unitary!")
    
    return U

def reunitarise(U):
    #Normalizing U
    U_norm = np.linalg.norm(U[0,:])
    U /= U_norm                  
    
    U[1,[1]] = U[0,[0]].conj()
    U[1,[0]] = -U[0,[1]].conj()
    
        
    return U