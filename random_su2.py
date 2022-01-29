import numpy as np
np.set_printoptions(precision=16)

sx = np.array(((0, 1),( 1, 0)))
sy = np.array(((0, -1j),(1j, 0)))
sz = np.array(((1, 0),(0, -1)))



def generate_su2_U(eps):
    su2_U = [] 
   
        
    r0  = np.random.uniform(-0.5,0.5)
    x0  = np.sign(r0)*np.sqrt(1-eps**2)
    
    r   = np.random.random((3)) - 0.5      
    x   = eps*r/np.linalg.norm(r)

    su2_U = x0*np.identity(2) + 1j*x[0]*sx + 1j*x[1]*sy + 1j*x[2]*sz  
    if (x0**2+np.linalg.norm(x)**2) != (1.0):
        print('U not normalized')
    
    return su2_U
    
