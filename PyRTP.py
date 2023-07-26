#%%
# Comments correct as 18/7/2023
# Commented by MT
# The purpose of these comments is to give an overview of the programming decisions, not the underlying physics.
# A Jupyter Notebook version of this program is available via [insert service here], which gives a short introduction
# to the physics underpinning the RT-TDDFT method.

#Package imports
import math # For general math operations
import numpy as np # For fast math operations using BLAS (i.e. sums, means, etc.)
import scipy # For general purpose maths, which sometimes works better than Numpy (FFTs specifically)
from scipy.interpolate import lagrange # Specifically importing the lagrange function to save space later on in the program
from sympy import Matrix # For general matrix manipulation (diagonalisation, etc). Numpy may work better for these operations, and may be a future change
import pylibxc # Python interpretter for the LibXC exchange-correlation potentials library
import os # Mostly used to create files and directories
from npy_append_array import NpyAppendArray #used to save data as external arrays regularly in case of failure to run
import matplotlib.pyplot as plt # Used for post-processing of data
import time # Used to check run times of certain functions
from numba import jit,njit,objmode # Translates Python script into machine code for significantly faster runtimes
import re # Used to read and format .txt and .dat files (reading basis set libraries, etc.)
from numba.typed import Dict
from numba.core import types
float_array = types.float64[:]

# the number of threads can be specified by uncommenting one of the following functions,
# run 'np.show_config()' to determine which to use.
#os.environ['OMP_NUM_THREADS']='8'
#os.environ['OPENBLAS_NUM_THREADS']='8'
#os.environ['MPI_NUM_THREADS']='8'
#os.environ['MKL_NUM_THREADS']='8'

#%%
# FunctionCalls
# This section includes all the functions required to perform DFT and RT-TDDFT. The @jit tags instruct Numba to convert
# these functions into machine code, and are cached, so that repeated runs do not have to do this translation on every run.
def GridCreate(L,N_i,functioncalls,timers):
    # This function takes the length of the grid and the number of points per side (N_i) and forms a 3-dimensional 'box' of
    # grid points. This function does not have the @jit tag, as Numba does not support the 'np.linspace' command natively.
    # This command also returns the total number of points and the grid spacing 'dr'.
    with objmode(st='f8'):
         st=time.time()
    N = N_i**3
    dr = (L/N_i)**3
    r_x = np.linspace(-L/2.,L/2.,int(N_i),endpoint=False)
    r_y = np.linspace(-L/2.,L/2.,int(N_i),endpoint=False)
    r_z = np.linspace(-L/2.,L/2.,int(N_i),endpoint=False)
    with objmode(et='f8'):
         et=time.time()
    functioncalls[0]+=1
    timers['GridCreatetimes']=np.append(timers['GridCreatetimes'],et-st)
    return r_x,r_y,r_z,N,dr,functioncalls,timers

@njit(cache=True)
def GTO(a,i,j,k,r,R) :

    x = r[0]-R[0]
    y = r[1]-R[1]
    z = r[2]-R[2]

    ans=(2*a/np.pi)**(3/4)*(((8*a)**(i+j+k)*math.gamma(i+1)*math.gamma(j+1)*math.gamma(k+1))/(math.gamma(2*i+1)*math.gamma(2*j+1)*math.gamma(2*k+1)))**(1/2)*x**i*y**j*z**k*np.exp(-a*(x**2+y**2+z**2))
    #math.gamma(n+1) = math.factorial(n) only gamma function is compatible with numba
    return ans
    
@njit(cache=True)
def construct_GTOs(orbital,N,N_i,r_x,r_y,r_z,R_I,Alpha,functioncalls,timers):

    with objmode(st='f8'):
        st=time.time()
    #create a matrix of grid points for each of the three GTOs, initialised to zero.
    GTO_p = np.zeros(len(Alpha)*N).reshape(len(Alpha),N_i,N_i,N_i)

    for gto in range(0,len(Alpha)) :
        for i in range(0,N_i) :
            for j in range(0,N_i) :
                for k in range(0,N_i) :
                    point = np.array([r_x[i],r_y[j],r_z[k]])

                    if orbital == 'S':
                        GTO_p[gto][i][j][k] = GTO(Alpha[gto],0,0,0,point,R_I)
                    if orbital == 'Px': 
                        GTO_p[gto][i][j][k] = GTO(Alpha[gto],1,0,0,point,R_I)
                    if orbital == 'Py': 
                        GTO_p[gto][i][j][k] = GTO(Alpha[gto],0,1,0,point,R_I)
                    if orbital == 'Pz': 
                        GTO_p[gto][i][j][k] = GTO(Alpha[gto],0,0,1,point,R_I)

    with objmode(et='f8'):
         et=time.time()
    functioncalls[1]+=1
    timers['constructGTOstimes']=np.append(timers['constructGTOstimes'],et-st)
    return GTO_p,functioncalls,timers

@njit(cache=True)
def construct_CGF(GTOs,N,N_i,Coef,functioncalls,timers):
    # This function evaluates the STO-3G approximation at each grid point.
    with objmode(st='f8'):
        st=time.time()
    CGF = np.zeros(N).reshape(N_i,N_i,N_i) #create a matrix of grid points initialised to zero.
    for g in range(0,len(GTOs)): 
        CGF += Coef[g]*GTOs[g]              #construct the CGF from the GTOs and coefficients, Eq. 2.
    with objmode(et='f8'):
         et=time.time()
    functioncalls[2]+=1
    timers['constructCGFtimes']=np.append(timers['constructCGFtimes'],et-st)
    return CGF,functioncalls,timers

def CP2K_basis_set_data(filename,element,basis_set,numorbitals,functioncalls,timers):
    # This function reads the selected basis set file within the CP2K data folder. In future, depending on licensing, we may be able to include a copy of the 'BASIS_MOLOPT' file, 
    # or read all of the data in the file and convert it into a .npy file.
    st=time.time()
    data_path = './src/BasisSets' # This folder contains the basis set .txt files
    check = 0
    get_key = 0
    key = np.array([])
    string_part = '(\d+?.\d+?|-\d+?.\d+?)\s+'

    alpha = np.array([])
    coef = np.array([])
    coef_temp = np.array([])

    with open(data_path+'/'+filename) as fin : # This sections loops over all lines and extracts the data required for the specified element
        for line in fin :
            if check == 1 :
                if line  == (' 1\n') :
                    get_key = 1
                    continue
                if get_key ==1 :
                    for string in line:
                        if string == '\n' : break
                        if string == ' '  : continue
                        else : key = np.append(key,int(string))
                    get_key = 0
                    if key[4:-1].size > 0  : coef_range = int(np.sum(key[4:-1]))
                    if key[4:-1].size == 0 : coef_range = int(key[4])
                    get_string = re.compile('\\s+'+string_part*(coef_range+1)+'\\+?') #+1 for exponents column                                                                                                                                       
                    continue
                match_string = get_string.match(line)
                if match_string :
                    alpha = np.append(alpha,match_string.groups()[0])
                    for i in range(0,coef_range) :
                        coef_temp = np.append(coef_temp,match_string.groups()[i+1])
                    if coef.size != 0 : coef = np.vstack([coef,coef_temp])
                    if coef.size == 0 : coef = coef_temp
                    coef_temp = np.array([])
                else : check = 0

            if line.startswith(' '+element+'  '+basis_set) or line.startswith(' '+element+' '+basis_set) :
                print(line)
                check = 1

    # Formatting data into useable Python variables
    coef = coef.T   
    #if key[2] > 0. : coef = coef.reshape(int(key[2]),int(coef_range/key[2]),int(key[3])) #l,function,data                                                                                                                                            
    #else : coef = coef.reshape(1,int(coef_range),int(key[3]))

    coef=np.array(coef,dtype=float).reshape(numorbitals,len(alpha))
    alpha=np.array(alpha,dtype=float)
    et=time.time()
    functioncalls[3]+=1
    timers['basissettimes']=np.append(timers['basissettimes'],et-st)
    return alpha.astype(np.float64),coef.astype(np.float64),functioncalls,timers

def normalise_basis_set(phi,dr):
    for u in range(0,len(phi)): 
        phi[u] = phi[u]*np.sqrt(1/np.sum(phi[u]*phi[u]*dr))
    
    return phi

@njit(cache=True)
def calculate_realspace_density(phi,N,N_i,P,dr,functioncalls,timers) :
    # This function determines the electron charge density at each grid point, along with a total electron charge density.
    with objmode(st='f8'):
        st=time.time()
    n_el_r = np.zeros(N).reshape(N_i,N_i,N_i)
    n_el_total = 0

    for i in range(0,len(phi)) :
        for j in range(0,len(phi)) :
            n_el_r += P[i][j]*phi[i]*phi[j]

    with objmode(et='f8'):
         et=time.time()
    functioncalls[4]+=1
    timers['calculaterealspacedensitytimes']=np.append(timers['calculaterealspacedensitytimes'],et-st)
    return n_el_r, np.sum(n_el_r)*dr,functioncalls,timers

@njit(cache=True)
def calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I,functioncalls,timers) :
    # This function determines the nuclear charge density at each grid point.
    with objmode(st='f8'):
        st=time.time()
    R_pp = np.sqrt(2.)/5. # This is a pseudopotential parameter, which is preset for each molecule.

    # It is worth mentioning at this point that this code is designed around simulating the HeH+ molecule. 
    # Further work is required to add the capability to simulate other molecules, which will have different
    # pseudopotential parameters.

    n_c_r = np.zeros(N).reshape(N_i,N_i,N_i)

    for i in range(0,N_i) :
        for j in range(0,N_i) :
            for k in range(0,N_i) :
                r = np.array([r_x[i],r_y[j],r_z[k]])

                for n in range(0,len(Z_I)) :
                    n_c_r[i][j][k] += -Z_I[n]/(R_pp**3)*np.pi**(-3/2)*np.exp(-((r-R_I[n])/R_pp).dot((r-R_I[n])/R_pp))
   
    with objmode(et='f8'):
         et=time.time()
    functioncalls[5]+=1
    timers['calculatecoredensitytimes']=np.append(timers['calculatecoredensitytimes'],et-st)
    return n_c_r,functioncalls,timers

@njit(cache=True,parallel=True)
def grid_integration(V_r,dr,phi,functioncalls,timers):
    # This function integrates an arbitrary potential over all points of the grid, giving a potential matrix in the
    # correct basis, which is more compact.
    with objmode(st='f8'):
        st=time.time()
    
    V = np.zeros(len(phi)**2).reshape(len(phi),len(phi))
    # Numba here
    for i in range(0,len(phi)):
        for j in range(0,len(phi)):

            V[i][j] = np.sum(np.real(V_r)*phi[i]*phi[j])*dr

    with objmode(et='f8'):
        et=time.time()
    functioncalls[6]+=1
    timers['gridintegrationtimes']=np.append(timers['gridintegrationtimes'],et-st)
    return V,functioncalls,timers

@njit(cache=True)
def energy_calculation(V,P,functioncalls,timers): 
    # This function evaluates a potential in the correct basis and determines the corresponding energy contribution. 
    with objmode(st='f8'):
        st=time.time()
    E=np.sum(P*V)
    with objmode(et='f8'):
         et=time.time()
    functioncalls[7]+=1
    timers['energycalculationtimes']=np.append(timers['energycalculationtimes'],et-st)
    return E,functioncalls,timers

@njit(cache=True)
def calculate_overlap(N,N_i,dr,phi,functioncalls,timers):
    # This function evaluates the overlap between atomic orbitals. It is worth noting that this function is simplified using
    # the grid integration function, as if the arbitrary potential being evaluated is equal to 1 at all points, the equations
    # to calculate the potential and the overlap are equivalent. Of course, this could be done explicitly in this function,
    # however it is more compact to do it this way.
    with objmode(st='f8'):
        st=time.time()
    V_temp = np.ones(N).reshape(N_i,N_i,N_i)
    S,functioncalls,timers = grid_integration(V_temp,dr,phi,functioncalls,timers)
    with objmode(et='f8'):
         et=time.time()
    functioncalls[8]+=1
    timers['calculateoverlaptimes']=np.append(timers['calculateoverlaptimes'],et-st)
    return S,functioncalls,timers

@njit(cache=True)
def calculate_kinetic_derivative(phi,phi_PW_G,N,N_i,G_u,G_v,G_w,L,functioncalls,timers) :
    # This function calculates the derivative of the kinetic energy by looping over all points of the reciprocal space grid.
    # 'with objmode()' is used to escape from the machine code to perform functions not available in Numba, with np.matmul being
    # somewhat awkward to use.
    with objmode(st='f8'):
        st=time.time()
    delta_T = np.zeros(len(phi)**2).reshape(len(phi),len(phi))

    for i in range(0,N_i) :
        for j in range(0,N_i) :
            for k in range(0,N_i) :

                g = np.array([G_u[i],G_v[j],G_w[k]])
                for I in range(0,len(phi)) :
                    for J in range(0,len(phi)):
                        with objmode():
                            delta_T[I][J] += 0.5*L**3/N**2*np.dot(g,g)*np.real(np.dot(np.conjugate(phi_PW_G[I][i][j][k]),phi_PW_G[J][i][j][k]))
    with objmode(et='f8'):
         et=time.time()
    functioncalls[9]+=1
    timers['calculatekineticderivativetimes']=np.append(timers['calculatekineticderivativetimes'],et-st)
    return delta_T,functioncalls,timers


@njit(cache=True)
def calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L,functioncalls,timers):
    # This function calculates the Hartree potential and energy in reciprocal space.
    with objmode(st='f8',nG='complex128[:,:,:]'):
        st=time.time()
        nG = np.fft.fftshift(n_G) #n_G is shifted to match same frequency domain as G (-pi,pi) instead of (0,2pi)
    Vg = np.zeros(N).reshape(N_i,N_i,N_i).astype(np.complex128)     # Arrays here are of type 'np.complex128', however we 
                                                                    # eventually ignore all non-real parts.

    E_hart_G = 0. ## Hartree energy in reciprocal space
    
    for i in range(0,N_i) :
        for j in range(0,N_i) :
            for k in range(0,N_i) :

                R_vec = np.array([r_x[i],r_y[j],r_z[k]]) # position vector in real space
                G_vec = np.array([G_u[i],G_v[j],G_w[k]]) # position vector in reciprocal space

                if np.dot(G_vec,G_vec) < 0.01 :  continue # Prevents a division by zero

                Vg[i][j][k] = 4*np.pi*nG[i][j][k]/np.dot(G_vec,G_vec)
                E_hart_G += np.conjugate(nG[i][j][k])*Vg[i][j][k] 
                
    E_hart_G *= L**3/N**2*0.5
    with objmode(et='f8',Vout='complex128[:,:,:]'):
         Vout=np.fft.ifftshift(Vg) #result is shifted back. 
         et=time.time()
    functioncalls[10]+=1
    timers['calculatehartreereciprocaltimes']=np.append(timers['calculatehartreereciprocaltimes'],et-st)
    return Vout, E_hart_G,functioncalls,timers 

@njit(cache=True)
def calculate_hartree_real(V_r,n_r,dr,functioncalls,timers) :
    with objmode(st='f8'):
        st=time.time()
    Har=0.5*np.sum(V_r*n_r)*dr
    with objmode(et='f8'):
         et=time.time()
    functioncalls[11]+=1
    timers['calculatehartreerealtimes']=np.append(timers['calculatehartreerealtimes'],et-st)
    return Har,functioncalls,timers


def calculate_XC_pylibxc(n_el_r,N_i,dr,functioncalls,timers):
    # This function uses the LibXC library to evaluate the exchange-correlation potential and energy 
    # over the entire grid. This requires the electron charge density to function. The XC functional
    # currently being used is 'LDA_XC_TETER93', which mimics the XC functional used in CP2K.
    with objmode(st='f8'):
        st=time.time()
    func=pylibxc.LibXCFunctional('LDA_XC_TETER93','unpolarized')
    inp={}
    inp["rho"]=n_el_r
    ret=func.compute(n_el_r)
    E_XC_r=ret['zk']
    V_XC_r=ret['vrho']
    V_XC_r = V_XC_r.reshape(N_i,N_i,N_i)
    E_XC_r = E_XC_r.reshape(N_i,N_i,N_i)
    E_XC=np.sum(E_XC_r*n_el_r)*dr
    with objmode(et='f8'):
         et=time.time()
    functioncalls[12]+=1
    timers['calculateXCtimes']=np.append(timers['calculateXCtimes'],et-st)
    return V_XC_r, E_XC,functioncalls,timers


@njit(cache=True)
def calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,alpha_PP,R_I,functioncalls,timers) :
    # This function calculates the short range pseudopotential, again by looping over all grid points.
    with objmode(st='f8'):
        st=time.time()
    V_SR_r = np.zeros(N).reshape(N_i,N_i,N_i)
    
    for i in range(0,len(V_SR_r)) :
        for j in range(0,len(V_SR_r)) :
            for k in range(0,len(V_SR_r)) :
                R_vec = np.array([r_x[i],r_y[j],r_z[k]])
                for n in range(0,len(Z_I)) : #loop over nuclei
                    r = np.linalg.norm(R_vec - R_I[n])
                    for c in range(0,len(Cpp)) : #loop over values of Cpp
                            V_SR_r[i][j][k] += Cpp[n][c]*(np.sqrt(2.)*alpha_PP[n]*r)**(2*(c+1)-2)*np.exp(-(alpha_PP[n]*r)**2) 

    with objmode(et='f8'):
         et=time.time()
    functioncalls[13]+=1
    timers['calculateVSRtimes']=np.append(timers['calculateVSRtimes'],et-st)     
    return V_SR_r,functioncalls,timers

@njit(cache=True)
def calculate_self_energy(Z_I,alpha_PP,functioncalls,timers) :
    # This function calculates the self energy correction term of the total energy, which is relatively simple.
    with objmode(st='f8'):
        st=time.time()
    R_pp = 1/alpha_PP #R_{I}^{c}
    self=np.sum(-(2*np.pi)**(-1/2)*Z_I**2/R_pp)
    with objmode(et='f8'):
         et=time.time()
    functioncalls[14]+=1
    timers['calculateselfenergytimes']=np.append(timers['calculateselfenergytimes'],et-st)
    return self,functioncalls,timers

@njit(cache=True)
def calculate_Ion_interaction(Z_I,R_I,alpha_PP,functioncalls,timers) :
    # This function calculates the energy from the interation between ions. 
    with objmode(st='f8'):
        st=time.time()
    E_II=0
    R_pp = 1/alpha_PP #R_{I}^{c}
    for i in range(0,len(Z_I)):
        for j in range(0,len(Z_I)):
            if i==j:
                continue
            else:
                E_II += Z_I[i]*Z_I[j]/np.linalg.norm(R_I[i]-R_I[j])*math.erfc(np.linalg.norm(R_I[i]-R_I[j])/np.sqrt(R_pp[i]**2+R_pp[j]**2))
    
    with objmode(et='f8'):
         et=time.time()
    functioncalls[15]+=1
    timers['calculateIItimes']=np.append(timers['calculateIItimes'],et-st)
    return E_II,functioncalls,timers

def P_init_guess_calc(T,S,N,N_i,Z_I,r_x,r_y,r_z,R_I,basis_filename,functioncalls,G_u,G_v,G_w,L,dr,phi,timers):
    # This function uses exclusively the core Hamiltonian to approximate the initial density matrix. This is then saved in the src/InitialGuesses directory
    # as a .npy file. When calling this function again, the program will check this directory for pre-calculated initial guesses, and if none are found, 
    # the function will compute it. It is worth noting that using the core Hamiltonian and the Hartree-Fock method is only suitable for small molecules.
    with objmode(st='f8'):
        st=time.time()
    Z_Iparts=Z_I.flatten().tolist()
    R_Iparts=R_I.flatten().tolist()
    Z_Istring = "".join([str(i) for i in Z_Iparts])
    R_Istring = "".join([str(i) for i in R_Iparts])
    
    filenamelist=[basis_filename,Z_Istring,R_Istring]
    filename="".join(filenamelist)
    if os.path.isfile("src/InitialGuesses/"+filename+".npy"):
        print("Reading P_init from src/InitialGuesses...\n")
        P_init = np.load("src/InitialGuesses/"+filename+".npy")
    else:
        print("Using HF to approximate P_init...\n")
        n_c_r,functioncalls,timers = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I,functioncalls,timers)
        n_c_G=np.fft.fftn(n_c_r)
        V_c_G, E_c_G,functioncalls,timers = calculate_hartree_reciprocal(n_c_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L,functioncalls,timers)
        V_c_r = np.fft.ifftn(V_c_G)
        V_nuc,functioncalls,timers = grid_integration(V_c_r,dr,phi,functioncalls,timers)
        H_core=T+V_nuc
        energies,C=scipy.linalg.eigh(H_core,S)
        P_init=np.zeros_like(C)
        for u in range(0,len(P_init)):
            for v in range(0,len(P_init)) :
                for p in range(0,len(P_init)-1) :
                    P_init[u][v] += C[u][p]*C[v][p]
                P_init[u][v] *=2
        np.save("src/InitialGuesses/"+filename,P_init)

    with objmode(et='f8'):
        et=time.time()
    functioncalls[29]+=1
    timers['P_initguesstimes']=np.append(timers['P_initguesstimes'],et-st)
    return P_init,functioncalls,timers

def PP_coefs(Z_I,functioncalls,timers):
    with objmode(st='f8'):
        st=time.time()
    cPP_lib=np.load('src/Pseudopotentials/Local/cPP.npy')
    rPP_lib=np.load('src/Pseudopotentials/Local/rPP.npy')
    cPP=[]
    rPP=[]
    for i in range(0,len(Z_I)):
        cPP.append(cPP_lib[int(Z_I[i])-1])
        rPP.append(rPP_lib[int(Z_I[i]-1)])

    alpha_PP=1/(np.sqrt(2)*np.array(rPP))
    with objmode(et='f8'):
        et=time.time()
    functioncalls[30]+=1
    timers['PP_coefstimes']=np.append(timers['PP_coefstimes'],et-st)
    return np.array(cPP).astype(np.float64),np.array(rPP).astype(np.float64),alpha_PP.astype(np.float64),functioncalls,timers

@njit
def spherical_harmonic(l,m,N,N_i,r_x,r_y,r_z) :

    #Y = np.complex128(np.zeros(N).reshape(N_i,N_i,N_i))
    Y = np.zeros(N,dtype=np.complex128).reshape(N_i,N_i,N_i)
    for i in range(0,N_i) :
        for j in range(0,N_i) :
            for k in range(0,N_i) :
                #theta,phi = cart_2_sph(r_x[i],r_y[j],r_z[k]) #convert to spherical coords
                #Y[i][j][k] = scipy.special.sph_harm(m,l,theta,phi) #harmonic function on the grid
                if l == 0 :                                                                                                                                                                                                                         
                    Y[i][j][k] = 0.5*np.sqrt(1/np.pi)                                                                                                                                                                                               
                    continue                                                                                                                                                                                                                        

                r = np.sqrt(r_x[i]**2+r_y[j]**2+r_z[k]**2)                                                                                                                                                                                          
                if r < 0.01 : continue                                                                                                                                                                                                              
                if m == -1: Y[i][j][k] = 0.5*np.sqrt(3/(2*np.pi))*(r_x[i]-np.sqrt(-1+0j)*r_y[j])/r                                                                                                                                                  
                if m == 0 : Y[i][j][k] = 0.5*np.sqrt(3/np.pi)*(r_z[k]/r)                                                                                                                                                                            
                if m == 1 : Y[i][j][k] = -0.5*np.sqrt(3/(2*np.pi))*(r_x[i]+np.sqrt(-1+0j)*r_y[j])/r         
                
    return Y

@njit
def projector(I,l,m,N,N_i,r_x,r_y,r_z,r_l,dr):

    p = np.zeros(N,dtype=np.complex128).reshape(N_i,N_i,N_i)
    tot = 0. #used for normalisation

    for i in range(0,N_i) :
        for j in range(0,N_i) :
            for k in range(0,N_i) :
                r = np.sqrt(r_x[i]**2+r_y[j]**2+r_z[k]**2)
                p[i][j][k] = r**(l+2*I-2)*np.exp(-0.5*(r/r_l[l])**2)
                tot += p[i][j][k]*p[i][j][k]*r**2*dr
    p = p/np.sqrt(tot) #normalisation
    
    return p

@njit
def pseudo_nl(phi,l,m,N,N_i,r_x,r_y,r_z,r_l,dr) :

    V_nl = np.zeros(len(phi)**2).reshape(len(phi),len(phi))
    
    for u in range(0,len(phi)) :
        for v in range(0,len(phi)) :
    
            for l in range(0,1) : #only s orbital pseudopotential in the calculation of H2O ?
                for i in range(1,2) : #only non-zero coefficient h is when i=j=1                                                                                                                                                                             
                    for j in range(1,2) :
                        for m in range(-l,l+1) :                                                                                                                                                                                                                

                            Y = spherical_harmonic(l,m,N,N_i,r_x,r_y,r_z)                                                                                                                                                                                                      

                            p_I = projector(i,l,m,N,N_i,r_x,r_y,r_z,r_l,dr)
                            p_J = projector(j,l,m,N,N_i,r_x,r_y,r_z,r_l,dr)
                            
                            #THIS IS NOT REALLY MATRIX MULTIPLICATION, JUST IN ARRAYS TO SPEED UP CODE (i.e. STILL POINT BY POINT MULTIPLICATION)                                                                                                                            
                            V_nl[u][v] += np.real(np.sum(np.conjugate(phi[u])*p_I*Y)*h_l[l]*np.sum(phi[v]*p_J*np.conjugate(Y))*dr*dr)
                 
    return V_nl

def element_interpreter(elements):
    Z_I=np.zeros(len(elements))
    orbitals=np.empty([len(elements),5],dtype=str)
    for i in range(0,len(elements)):
        match elements[i]:
            case 'H':
                Z_I[i]=1.0
                orbitals[i]=['S','','','','']
            case 'He':
                Z_I[i]=2.0
                orbitals[i]=['S','','','','']
            case 'Li':
                Z_I[i]=3.0
                orbitals[i]=['S','S','','','']
            case 'Be':
                Z_I[i]=4.0
                orbitals[i]=['S','S','','','']
            case 'B':
                Z_I[i]=5.0
                orbitals[i]=['S','Px','Py','Pz','']
            case 'C':
                Z_I[i]=6.0
                orbitals[i]=['S','Px','Py','Pz','']
            case 'N':
                Z_I[i]=7.0
                orbitals[i]=['S','Px','Py','Pz','']
            case 'O':
                Z_I[i]=8.0
                orbitals[i]=['S','Px','Py','Pz','']
            case 'F':
                Z_I[i]=9.0
                orbitals[i]=['S','Px','Py','Pz','']
            case 'Ne':
                Z_I[i]=10.0
                orbitals[i]=['S','Px','Py','Pz','']
            case 'Na':
                Z_I[i]=11.0
                orbitals[i]=['S','Px','Py','Pz','S']
            case 'Mg':
                Z_I[i]=12.0
                orbitals[i]=['S','Px','Py','Pz','S']
            case 'Al':
                Z_I[i]=13.0
                orbitals[i]=['S','Px','Py','Pz','']
            case 'Si':
                Z_I[i]=14.0
                orbitals[i]=['S','Px','Py','Pz','']
            case 'P':
                Z_I[i]=15.0
                orbitals[i]=['S','Px','Py','Pz','']
            case 'S':
                Z_I[i]=16.0
                orbitals[i]=['S','Px','Py','Pz','']
            case 'Cl':
                Z_I[i]=17.0
                orbitals[i]=['S','Px','Py','Pz','']
            case 'Ar':
                Z_I[i]=18.0
                orbitals[i]=['S','Px','Py','Pz','']
            case _:
                raise TypeError("element["+str(i)+"] is invalid")
    
    tot_orbitals=0
    for i in range(0,len(elements)):
        for j in range(0,np.size(orbitals,1)):
            if orbitals[i][j]!='':
                tot_orbitals+=1

    return np.array(Z_I),orbitals,tot_orbitals

def dftSetup(R_I,L,N_i,elements,basis_sets,basis_filename,functioncalls,timers):
    # This function calls all functions required to perform DFT that do not change during the SCF cycle.
    # This prevents the functions being called unneccessarily. This function should be called prior to the 
    # 'computeDFT' function.

    with objmode(st='f8'):
        st=time.time()
    Z_I,orbitals,tot_orbitals=element_interpreter(elements)
    r_x,r_y,r_z,N,dr,functioncalls,timers=GridCreate(L,N_i,functioncalls,timers)
    phi = np.zeros(tot_orbitals*N).reshape(tot_orbitals,N_i,N_i,N_i)
    k=0
    
    for i in range(0,len(elements)):
        l=0
        alpha_temp,coef_temp,functioncalls,timers = CP2K_basis_set_data(basis_filename,elements[i],basis_sets[i],np.count_nonzero(orbitals[i]!=''),functioncalls,timers)
        for j in range(0,len(orbitals[i])):
            if orbitals[i][j]=='':
                continue
            else:
                GTOs_temp,functioncalls,timers = construct_GTOs(orbitals[i][j],N,N_i,r_x,r_y,r_z,R_I[i],alpha_temp,functioncalls,timers)
                phi[k],functioncalls,timers = construct_CGF(GTOs_temp,N,N_i,coef_temp[l],functioncalls,timers)
                l+=1
                k+=1
                

    phi=normalise_basis_set(phi,dr)
    print(phi.shape)
    G_u = np.linspace(-N_i*np.pi/L,N_i*np.pi/L,N_i,endpoint=False)
    G_v = np.linspace(-N_i*np.pi/L,N_i*np.pi/L,N_i,endpoint=False)
    G_w = np.linspace(-N_i*np.pi/L,N_i*np.pi/L,N_i,endpoint=False)
    phi_PW_G = []
    for i in range(0,len(phi)):
        phi_PW_G.append(np.fft.fftshift(np.fft.fftn(phi[i]))) #PW representation for He in reciprocal space, G #PW representation for H in reciprocal space, G
    phi_PW_G = np.array(phi_PW_G) #store in array 
    S, functioncalls,timers = calculate_overlap(N,N_i,dr,phi,functioncalls,timers)
    delta_T,functioncalls,timers = calculate_kinetic_derivative(phi,phi_PW_G,N,N_i,G_u,G_v,G_w,L,functioncalls,timers)
    cPP,rPP,alpha_PP,functioncalls,timers=PP_coefs(Z_I,functioncalls,timers)
    E_self,functioncalls,timers = calculate_self_energy(Z_I,alpha_PP,functioncalls,timers)
    E_II,functioncalls,timers = calculate_Ion_interaction(Z_I,R_I,alpha_PP,functioncalls,timers)
    with objmode(et='f8'):
         et=time.time()
    functioncalls[16]+=1
    timers['DFTsetuptimes']=np.append(timers['DFTsetuptimes'],et-st)
    return Z_I,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,[],[],S,delta_T,E_self,E_II,phi,cPP,rPP,alpha_PP,functioncalls,timers


def computeE_0(R_I,Z_I,P,N_i,Cpp,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,rPP,alpha_PP,functioncalls,timers):
    # This function returns the total energy for a given density matrix P.
    with objmode(st='f8'):
        st=time.time()
    n_el_r, n_el_r_tot,functioncalls,timers  = calculate_realspace_density(phi,N,N_i,P,dr,functioncalls,timers)
    n_c_r,functioncalls,timers = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I,functioncalls,timers)
    n_r = n_el_r + n_c_r
    T,functioncalls,timers = energy_calculation(delta_T,P,functioncalls,timers)
    n_G = np.fft.fftn(n_r)
    V_G, E_hart_G,functioncalls,timers = calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L,functioncalls,timers)
    V_r = np.fft.ifftn(V_G)
    V_hart,functioncalls,timers = grid_integration(V_r,dr,phi,functioncalls,timers)
    E_hart_r,functioncalls,timers = calculate_hartree_real(V_r,n_r,dr,functioncalls,timers)
    V_XC_r,E_XC,functioncalls,timers = calculate_XC_pylibxc(n_el_r,N_i,dr,functioncalls,timers)
    V_XC,functioncalls,timers = grid_integration(V_XC_r,dr,phi,functioncalls,timers)
    V_SR_r,functioncalls,timers = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,alpha_PP,R_I,functioncalls,timers)
    V_SR,functioncalls,timers = grid_integration(V_SR_r,dr,phi,functioncalls,timers)
    E_SR,functioncalls,timers = energy_calculation(V_SR,P,functioncalls,timers)
    E_0 = E_hart_r + E_XC + E_SR + T + E_self + E_II
    with objmode(et='f8'):
         et=time.time()
    functioncalls[17]+=1
    timers['computeE0times']=np.append(timers['computeE0times'],et-st)
    return E_0,functioncalls,timers

def computeDFT_first(R_I,L,N_i,Z_I,Cpp,iterations,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,S,delta_T,E_self,E_II,phi,rPP,alpha_PP,basis_filename,functioncalls,timers):
    # This function performs conventional ground-state DFT to produce a converged ground state electron density, wavefunction, energy and Kohn-Sham orbitals.
    # The 'DFTsetup' function must be called prior to calling this function.
    with objmode(st='f8'):
        st=time.time()
    P_init,functioncalls,timers = P_init_guess_calc(delta_T,S,N,N_i,Z_I,r_x,r_y,r_z,R_I,basis_filename,functioncalls,G_u,G_v,G_w,L,dr,phi,timers)
    P=P_init
    n_el_r, n_el_r_tot,functioncalls,timers  = calculate_realspace_density(phi,N,N_i,P,dr,functioncalls,timers)
    n_c_r,functioncalls,timers = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I,functioncalls,timers)
    n_r = n_el_r + n_c_r
    T,functioncalls,timers = energy_calculation(delta_T,P,functioncalls,timers)
    n_G = np.fft.fftn(n_r)
    V_G, E_hart_G,functioncalls,timers = calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L,functioncalls,timers)
    V_r = np.fft.ifftn(V_G)
    V_hart,functioncalls,timers = grid_integration(V_r,dr,phi,functioncalls,timers)
    E_hart_r,functioncalls,timers = calculate_hartree_real(V_r,n_r,dr,functioncalls,timers)
    V_XC_r,E_XC,functioncalls,timers = calculate_XC_pylibxc(n_el_r,N_i,dr,functioncalls,timers)
    V_XC,functioncalls,timers = grid_integration(V_XC_r,dr,phi,functioncalls,timers)
    V_SR_r,functioncalls,timers = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,alpha_PP,R_I,functioncalls,timers)
    V_SR,functioncalls,timers = grid_integration(V_SR_r,dr,phi,functioncalls,timers)
    E_SR,functioncalls,timers = energy_calculation(V_SR,P,functioncalls,timers)
    KS = np.array(delta_T)+np.array(V_hart)+np.array(V_SR)+np.array(V_XC)
    KS = np.real(KS) #change data type from complex to float, removing all ~0. complex values
    S=Matrix(S)
    U,s = S.diagonalize()
    s = s**(-0.5)
    X = np.matmul(np.array(U,dtype='float64'),np.array(s,dtype='float64'))
    X_dag = np.matrix.transpose(np.array(X,dtype='float64'))
    err = 1.0e-6 #The error margin by which convergence of the P matrix is measured

    P = P_init #reset P to atomic guess.
    for I in range(0,iterations):
        n_el_r, n_el_r_tot,functioncalls,timers  = calculate_realspace_density(phi,N,N_i,P,dr,functioncalls,timers)
        n_c_r,functioncalls,timers = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I,functioncalls,timers)
        n_r = n_el_r + n_c_r
        n_G = np.fft.fftn(n_r)
        V_G, E_hart_G,functioncalls,timers = calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L,functioncalls,timers)
        V_r = np.fft.ifftn(V_G)
        V_hart,functioncalls,timers = grid_integration(V_r,dr,phi,functioncalls,timers)
        E_hart_r,functioncalls,timers = calculate_hartree_real(V_r,n_r,dr,functioncalls,timers)
        V_XC_r,E_XC,functioncalls,timers = calculate_XC_pylibxc(n_el_r,N_i,dr,functioncalls,timers)
        V_XC,functioncalls,timers = grid_integration(V_XC_r,dr,phi,functioncalls,timers)
        V_SR_r,functioncalls,timers = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,alpha_PP,R_I,functioncalls,timers)
        V_SR,functioncalls,timers = grid_integration(V_SR_r,dr,phi,functioncalls,timers)
        E_SR,functioncalls,timers = energy_calculation(V_SR,P,functioncalls,timers)
        E_0 = E_hart_r + E_XC + E_SR + T + E_self + E_II
        KS = np.array(delta_T)+np.array(V_hart)+np.array(V_SR)+np.array(V_XC)
        KS = np.real(KS)
        KS_temp = Matrix(np.matmul(X_dag,np.matmul(KS,X)))
        C_temp, e = KS_temp.diagonalize()
        C = np.matmul(X,C_temp)
        print("iteration : ", I+1, "\n")               # This outputs various information for the user to monitor
        print("total energy : ", np.real(E_0), "\n")   # the convergence of the SCF cycle. These may be commented out
        print("density matrix : ","\n", P, "\n")       # if required.
        print("KS matrix : ","\n", KS, "\n")
        P_new=np.zeros_like(P_init)
        for u in range(0,2):
            for v in range(0,2) :
                for p in range(0,1) :
                    P_new[u][v] += C[u][p]*C[v][p]
                P_new[u][v] *=2
        if np.all(abs(P-P_new)<= err)==True:
            break
                
        P = P_new 
    with objmode(et='f8'):
         et=time.time()
    functioncalls[18]+=1
    timers['computeDFTtimes']=np.append(timers['computeDFTtimes'],et-st)
    return P,np.real(E_0),C,KS,functioncalls,timers

def computeDFT(R_I,L,N_i,P_init,Z_I,Cpp,iterations,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi,rPP,alpha_PP,functioncalls,timers):
    # This function performs conventional ground-state DFT to produce a converged ground state electron density, wavefunction, energy and Kohn-Sham orbitals.
    # The 'DFTsetup' function must be called prior to calling this function.
    with objmode(st='f8'):
        st=time.time()
    P = P_init
    n_el_r, n_el_r_tot,functioncalls,timers  = calculate_realspace_density(phi,N,N_i,P,dr,functioncalls,timers)
    n_c_r,functioncalls,timers = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I,functioncalls,timers)
    n_r = n_el_r + n_c_r
    T,functioncalls,timers = energy_calculation(delta_T,P,functioncalls,timers)
    n_G = np.fft.fftn(n_r)
    V_G, E_hart_G,functioncalls,timers = calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L,functioncalls,timers)
    V_r = np.fft.ifftn(V_G)
    V_hart,functioncalls,timers = grid_integration(V_r,dr,phi,functioncalls,timers)
    E_hart_r,functioncalls,timers = calculate_hartree_real(V_r,n_r,dr,functioncalls,timers)
    V_XC_r,E_XC,functioncalls,timers = calculate_XC_pylibxc(n_el_r,N_i,dr,functioncalls,timers)
    V_XC,functioncalls,timers = grid_integration(V_XC_r,dr,phi,functioncalls,timers)
    V_SR_r,functioncalls,timers = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,alpha_PP,R_I,functioncalls,timers)
    V_SR,functioncalls,timers = grid_integration(V_SR_r,dr,phi,functioncalls,timers)
    E_SR,functioncalls,timers = energy_calculation(V_SR,P,functioncalls,timers)
    KS = np.array(delta_T)+np.array(V_hart)+np.array(V_SR)+np.array(V_XC)
    KS = np.real(KS) #change data type from complex to float, removing all ~0. complex values
    S=Matrix(S)
    U,s = S.diagonalize()
    s = s**(-0.5)
    X = np.matmul(np.array(U,dtype='float64'),np.array(s,dtype='float64'))
    X_dag = np.matrix.transpose(np.array(X,dtype='float64'))
    err = 1.0e-6 #The error margin by which convergence of the P matrix is measured

    P = P_init #reset P to atomic guess.
    for I in range(0,iterations):
        n_el_r, n_el_r_tot,functioncalls,timers  = calculate_realspace_density(phi,N,N_i,P,dr,functioncalls,timers)
        n_c_r,functioncalls,timers = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I,functioncalls,timers)
        n_r = n_el_r + n_c_r
        n_G = np.fft.fftn(n_r)
        V_G, E_hart_G,functioncalls,timers = calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L,functioncalls,timers)
        V_r = np.fft.ifftn(V_G)
        V_hart,functioncalls,timers = grid_integration(V_r,dr,phi,functioncalls,timers)
        E_hart_r,functioncalls,timers = calculate_hartree_real(V_r,n_r,dr,functioncalls,timers)
        V_XC_r,E_XC,functioncalls,timers = calculate_XC_pylibxc(n_el_r,N_i,dr,functioncalls,timers)
        V_XC,functioncalls,timers = grid_integration(V_XC_r,dr,phi,functioncalls,timers)
        V_SR_r,functioncalls,timers = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,alpha_PP,R_I,functioncalls,timers)
        V_SR,functioncalls,timers = grid_integration(V_SR_r,dr,phi,functioncalls,timers)
        E_SR,functioncalls,timers = energy_calculation(V_SR,P,functioncalls,timers)
        E_0 = E_hart_r + E_XC + E_SR + T + E_self + E_II
        KS = np.array(delta_T)+np.array(V_hart)+np.array(V_SR)+np.array(V_XC)
        KS = np.real(KS)
        KS_temp = Matrix(np.matmul(X_dag,np.matmul(KS,X)))
        C_temp, e = KS_temp.diagonalize()
        C = np.matmul(X,C_temp)
        print("iteration : ", I+1, "\n")               # This outputs various information for the user to monitor
        print("total energy : ", np.real(E_0), "\n")   # the convergence of the SCF cycle. These may be commented out
        print("density matrix : ","\n", P, "\n")       # if required.
        print("KS matrix : ","\n", KS, "\n")
        P_new=np.zeros_like(P_init)
        for u in range(0,2):
            for v in range(0,2) :
                for p in range(0,1) :
                    P_new[u][v] += C[u][p]*C[v][p]
                P_new[u][v] *=2
        if np.all(abs(P-P_new)<= err)==True:
            break
                
        P = P_new 
    with objmode(et='f8'):
         et=time.time()
    functioncalls[18]+=1
    timers['computeDFTtimes']=np.append(timers['computeDFTtimes'],et-st)
    return P,np.real(E_0),C,KS,functioncalls,timers

# GaussianKick - provides energy to the system in the form of an electric pulse
# with a Gaussian envelop (run each propagation step!)
def GaussianKick(KS,scale,direction,t,r_x,r_y,r_z,phi,dr,functioncalls,timers):
    with objmode(st='f8'):
        st=time.time()

    t0=1 # These parameters, representing the pulse center and width
    w=0.2 # were selected to be ideal for most systems.
             # In future, it is possible that the user could be given some
             # control over these parameters.

    Efield=np.dot(scale*np.exp((-(t-t0)**2)/(2*(w**2))),direction)
    D_x,D_y,D_z,D_tot,functioncalls,timers=transition_dipole_tensor_calculation(r_x,r_y,r_z,phi,dr,functioncalls,timers)
    V_app=-(np.dot(D_x,Efield[0])+np.dot(D_y,Efield[1])+np.dot(D_z,Efield[2]))
    KS_new=KS+V_app
    with objmode(et='f8'):
         et=time.time()
    functioncalls[19]+=1
    timers['GaussianKicktimes']=np.append(timers['GaussianKicktimes'],et-st)
    return KS_new,functioncalls,timers

# Propagator - this function selects the equation to determine the unitary operator to propagate 
# the system forward in time a singular timestep, using predictor-corrector regime (if applicable)
# This function uses a match-case statement, hence the need for Python 3.10 or greater.
def propagator(select,H1,H2,dt,functioncalls,timers): #propagator functions
    with objmode(st='f8'):
        st=time.time()
    match select:
        case 'CN':
            U=(1-(1j*(dt/2)*H1))/(1+(1j*(dt/2)*H1))
        case 'EM':
            U=np.exp(-1j*dt*H1)
        case 'ETRS':
            U=np.exp(-1j*(dt/2)*H2)*np.exp(-1j*(dt/2)*H1)
        case 'AETRS':
            U=np.exp(-1j*(dt/2)*H2)*np.exp(-1j*(dt/2)*H1)
        case 'CAETRS':
            U=np.exp(-1j*(dt/2)*H2)*np.exp(-1j*(dt/2)*H1)
        case 'CFM4':
            a1=(3-2*np.sqrt(3))/12
            a2=(3+2*np.sqrt(3))/12
            U=np.dot(np.exp(-1j*dt*a1*H1-1j*dt*a2*H2),np.exp(-1j*dt*a2*H1-1j*dt*a1*H2))
        case _:
            raise TypeError("Invalid propagator")
    with objmode(et='f8'):
        et=time.time()
    functioncalls[20]+=1
    timers['propagatortimes']=np.append(timers['propagatortimes'],et-st)
    return U,functioncalls,timers

def propagate(R_I,Z_I,P,H,C,dt,select,N_i,Cpp,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,delta_T,E_self,E_II,L,Hprev,t,energies,tnew,i,phi,rPP,alpha_PP,functioncalls,timers):
    # This function performs the propagation to determine the new density matrix at the next timestep, depending on the propagator selected.
    match select:
        case 'CN':
            # The Crank-Nicholson propagator uses a predictor-corrector algorithm to ensure the propagation is accurate.
            with objmode(st='f8'):
                st=time.time()
            # This is the predicting section
            if i==0:
                H_p=H
            elif i>0 and i<5:
                H_p,functioncalls,timers=LagrangeExtrapolate(t[0:i],energies[0:i],t[i-1]+(1/2),functioncalls,timers)  # LagrangeExtrapolate fits a curve to the previous set of energies
                                                                                                                                                          # and extrapolates based on that. It is important that the number of
            else:                                                                                                                                         # points be limited so as not to cause instability.
                H_p,functioncalls,timers=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],t[i-1]+(1/2),functioncalls,timers)  

            U,functioncalls,timers=propagator(select,H_p,[],dt,functioncalls,timers) # calling the propagtor function to determine the unitary operator U.
            U=np.real(U)
            C_p=np.dot(U,C)
            # This is the updating section
            P_p=np.zeros_like(P)
            for u in range(0,len(P_p)):
                for v in range(0,len(P_p)) :
                    for p in range(0,len(P_p)-1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0,functioncalls,timers=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,rPP,alpha_PP,functioncalls,timers)
            # This is the correcting section
            H_c=H+(1/2)*(E_0-H)
            U,functioncalls,timers=propagator(select,H_c,[],dt,functioncalls,timers) # After the predictor-corrector regime, the unitary operator U is likely more accurate.
            U=np.real(U)
            C_new=np.dot(U,C)
            P_new=np.zeros_like(P)
            for u in range(0,len(P_new)):
                for v in range(0,len(P_new)) :
                    for p in range(0,len(P_new)-1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            with objmode(et='f8'):
                et=time.time()
        case 'EM':
            # The Exponential Midpoint propagator uses a predictor-corrector algorithm to ensure the propagation is accurate.
            with objmode(st='f8'):
                st=time.time()
            # This is the predicting section
            if i==0:
                H_p=H
            elif i>0 and i<5:
                H_p,functioncalls,timers=LagrangeExtrapolate(t[0:i],energies[0:i],t[i-1]+(1/2),functioncalls,timers)  
            else:
                H_p,functioncalls,timers=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],t[i-1]+(1/2),functioncalls,timers)    
            U,functioncalls,timers=propagator(select,H_p,[],dt,functioncalls,timers)
            U=np.real(U)
            C_p=np.dot(U,C)
            # This is the updating section
            P_p=np.zeros_like(P)
            for u in range(0,len(P_p)):
                for v in range(0,len(P_p)) :
                    for p in range(0,len(P_p)-1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0,functioncalls,timers=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,rPP,alpha_PP,functioncalls,timers)
            # This is the correcting section
            H_c=H+(1/2)*(E_0-H)
            U,functioncalls,timers=propagator(select,H_c,[],dt,functioncalls,timers)
            U=np.real(U)
            C_new=np.dot(U,C)
            P_new=np.zeros_like(P)
            for u in range(0,len(P_new)):
                for v in range(0,len(P_new)) :
                    for p in range(0,len(P_new)-1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            with objmode(et='f8'):
                et=time.time()
        case 'ETRS':
            # The ETRS propagator requires an approximation of the energy at the next timestep. As such,
            # the EM propagator is used first to determine the energy at the next timestep, then the ETRS
            # propagator is used to ensure the density matrix is accurate and time-reversal symmetry is enforced.
            with objmode(st='f8'):
                st=time.time()
            # This is the predicting section of the EM propagator
            if i==0:
                H_p=H
            elif i>0 and i<5:
                H_p,functioncalls,timers=LagrangeExtrapolate(t[0:i],energies[0:i],t[i-1]+(1/2),functioncalls,timers) 
            else:
                H_p,functioncalls,timers=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],t[i-1]+(1/2),functioncalls,timers)     
            U,functioncalls,timers=propagator('EM',H_p,[],dt,functioncalls,timers)
            U=np.real(U)
            C_p=np.dot(U,C)
            # This is the updating section of the EM propagator
            P_p=np.zeros_like(P)
            for u in range(0,len(P_p)):
                for v in range(0,len(P_p)) :
                    for p in range(0,len(P_p)-1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0,functioncalls,timers=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,rPP,alpha_PP,functioncalls,timers)
            # This is the correcting section of the EM propagator
            H_c=H+(1/2)*(E_0-H)
            U,functioncalls,timers=propagator('EM',H_c,[],dt,functioncalls,timers)
            U=np.real(U)
            C_new=np.dot(U,C)
            P_new=np.zeros_like(P)
            for u in range(0,len(P_new)):
                for v in range(0,len(P_new)) :
                    for p in range(0,len(P_new)-1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            # As the EM propagator has determined a good approximate of the density matrix at the next timestep,
            # the 'computeE_0' function can be called to determine the energy at the next timestep.
            Hdt,functioncalls,timers=computeE_0(R_I,Z_I,P_new,N_i,Cpp,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,rPP,alpha_PP,functioncalls,timers)
            # The ETRS propagator can now be called
            U,functioncalls,timers=propagator(select,H,Hdt,dt,functioncalls,timers)
            U=np.real(U)
            C_new=np.dot(U,C)
            P_new=np.zeros_like(P)
            for u in range(0,len(P_new)):
                for v in range(0,len(P_new)) :
                    for p in range(0,len(P_new)-1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            with objmode(et='f8'):
                et=time.time()
        case 'AETRS':
            # The AETRS propagator uses a similar mechanism to the ETRS propagator, except rather than using the EM
            # propagator to predict the new density matrix (and thus energy), Lagrange Extrapolation is used.
            with objmode(st='f8'):
                st=time.time()
            if i<5:
                Hdt,functioncalls,timers=LagrangeExtrapolate(t[0:i],energies[0:i],tnew,functioncalls,timers) 
            else:
                Hdt,functioncalls,timers=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],tnew,functioncalls,timers)   
            U,functioncalls,timers=propagator(select,H,Hdt,dt,functioncalls,timers)
            U=np.real(U)
            C_new=np.dot(U,C)
            P_new=np.zeros_like(P)
            for u in range(0,len(P_new)):
                for v in range(0,len(P_new)) :
                    for p in range(0,len(P_new)-1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            with objmode(et='f8'):
                et=time.time()
        case 'CAETRS':
            # The CAETRS propagator works much in the same way as the AETRS propagator, with the inclusion of a
            # predictor-corrector algorithm to ensure accuracy, whilst still being significantly more efficient 
            # than the ETRS propagator.
            with objmode(st='f8'):
                st=time.time()    
            # This is the predicting section
            if i<5:
                Hdt,functioncalls,timers=LagrangeExtrapolate(t[0:i],energies[0:i],tnew,functioncalls,timers) 
            else:
                Hdt,functioncalls,timers=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],tnew,functioncalls,timers) 
            # This is the updating section
            U,functioncalls,timers=propagator(select,H,Hdt,dt,functioncalls,timers)
            U=np.real(U)
            C_p=np.dot(U,C)
            P_p=np.zeros_like(P)
            for u in range(0,len(P_p)):
                for v in range(0,len(P_p)) :
                    for p in range(0,len(P_p)-1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0,functioncalls,timers=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,rPP,alpha_PP,functioncalls,timers)
            # This is the correcting section
            H_c=H+(1/2)*(E_0-H)
            Hdt=2*H_c-Hprev
            U,functioncalls,timers=propagator(select,H_c,Hdt,dt,functioncalls,timers)
            U=np.real(U)
            C_new=np.dot(U,C)
            P_new=np.zeros_like(P)
            for u in range(0,len(P_new)):
                for v in range(0,len(P_new)) :
                    for p in range(0,len(P_new)-1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            with objmode(et='f8'):
                et=time.time()
        case 'CFM4':
            # The commutator-free 4th order Magnus propagator uses 2 approximations of energies at future times to
            # provide an accurate and efficient approximation of the unitary operator. This method is preferred.
            with objmode(st='f8'):
                st=time.time()
            if i<5:
                Ht1,functioncalls,timers=LagrangeExtrapolate(t[0:i],energies[0:i],(t[i-1])+((1/2)-(np.sqrt(3)/6)),functioncalls,timers)
                Ht2,functioncalls,timers=LagrangeExtrapolate(t[0:i],energies[0:i],(t[i-1])+((1/2)+(np.sqrt(3)/6)),functioncalls,timers) 
            else:
                Ht1,functioncalls,timers=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],(t[i-1])+((1/2)-(np.sqrt(3)/6)),functioncalls,timers)
                Ht2,functioncalls,timers=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],(t[i-1])+((1/2)+(np.sqrt(3)/6)),functioncalls,timers) 
            
            U,functioncalls,timers=propagator(select,Ht1,Ht2,dt,functioncalls,timers)
            U=np.real(U)
            C_new=np.dot(U,C)
            P_new=np.zeros_like(P)
            for u in range(0,len(P_new)):
                for v in range(0,len(P_new)) :
                    for p in range(0,len(P_new)-1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            with objmode(et='f8'):
                et=time.time()
        case _:
            raise TypeError("Invalid propagator")
    functioncalls[21]+=1
    timers['propagatetimes']=np.append(timers['propagatetimes'],et-st)
    return P_new,et-st,functioncalls,timers

@njit(cache=True)
def transition_dipole_tensor_calculation(r_x,r_y,r_z,phi,dr,functioncalls,timers) :
    # This function determines the transition dipole tensor in the x, y and z directions.
    # This is used for calculation of the Gaussian pulse and the dipole moment.
    with objmode(st='f8'):
        st=time.time()
    D_x,functioncalls,timers = grid_integration(r_x,dr,phi,functioncalls,timers)             
    D_y,functioncalls,timers = grid_integration(r_y,dr,phi,functioncalls,timers)
    D_z,functioncalls,timers = grid_integration(r_z,dr,phi,functioncalls,timers)   

    D_tot=D_x+D_y+D_z
    with objmode(et='f8'):
         et=time.time()
    functioncalls[22]+=1
    timers['tdttimes']=np.append(timers['tdttimes'],et-st)
    return D_x,D_y,D_z,D_tot,functioncalls,timers

def GetP(KS,S,functioncalls,timers):
    # This function determines the density matrix for a given Kohn-Sham matrix.
    with objmode(st='f8'):
        st=time.time()
    S=Matrix(S)
    U,s = S.diagonalize()
    s = s**(-0.5)
    X = np.matmul(np.array(U,dtype='float64'),np.array(s,dtype='float64'))
    X_dag = np.matrix.transpose(np.array(X,dtype='float64'))
    KS_temp = Matrix(np.matmul(X_dag,np.matmul(KS,X)))
    C_temp, e = KS_temp.diagonalize()
    C = np.matmul(X,C_temp)
    P_new=np.zeros_like(KS)
    for u in range(0,len(P_new)):
        for v in range(0,len(P_new)) :
            for p in range(0,len(P_new)-1) :
                P_new[u][v] += C[u][p]*C[v][p]
            P_new[u][v] *=2
    with objmode(et='f8'):
         et=time.time()
    functioncalls[23]+=1
    timers['getptimes']=np.append(timers['getptimes'],et-st)
    return P_new,functioncalls,timers

def LagrangeExtrapolate(t,H,tnew,functioncalls,timers):
    # This function performs Lagrange extrapolation (see Scipy documentation for details).
    with objmode(st='f8'):
        st=time.time()
    f=lagrange(t,H)
    val=np.real(f(tnew))
    with objmode(et='f8'):
         et=time.time()
    functioncalls[24]+=1
    timers['LagrangeExtrapolatetimes']=np.append(timers['LagrangeExtrapolatetimes'],et-st)
    return val,functioncalls,timers

@njit(cache=True)
def pop_analysis(P,S,functioncalls,popanalysistimes) :
    # This function provides Mulliken population analysis for HeH+, however this function will break for larger molecules.
    with objmode(st='f8'):
        st=time.time()
    PS = np.matmul(P,S)
    pop_total = np.trace(PS)
    pop_He = PS[0,0]
    pop_H = PS[1,1]
    with objmode(et='f8'):
         et=time.time()
    functioncalls[25]+=1
    popanalysistimes=np.append(popanalysistimes,et-st)
    return pop_total, pop_He, pop_H,functioncalls,popanalysistimes

def ShannonEntropy(P,functioncalls,timers):
    # This function determines the Shannon entropy, which has a variety of quantum chemistry uses.
    with objmode(st='f8'):
        st=time.time()
    P_eigvals=np.linalg.eigvals(P)
    P_eigvals_corrected=[x for x in P_eigvals if x>0.00001] # The eigenvalues are tidied to remove small values, as
    P_eigvecbasis=np.diag(P_eigvals_corrected)              # computing the natural log of small values gives inf.
    SE=np.trace(np.dot(P_eigvecbasis,np.log(P_eigvecbasis)))
    with objmode(et='f8'):
         et=time.time()
    functioncalls[26]+=1
    timers['SEtimes']=np.append(timers['SEtimes'],et-st)
    return SE,functioncalls,timers

def vonNeumannEntropy(P,functioncalls,timers):
    # This function determines the von Neumann entropy, which represents the 'distance' from a pure state in this program.
    with objmode(st='f8'):
        st=time.time()
    vNE=np.trace(np.dot(P,np.log(P)))
    with objmode(et='f8'):
         et=time.time()
    functioncalls[27]+=1
    timers['vNEtimes']=np.append(timers['vNEtimes'],et-st)
    return vNE,functioncalls,timers

def EntropyPInitBasis(P,Pini,functioncalls,timers):
    # This function produces an entropy-like measure based on the initial density matrix.
    with objmode(st='f8'):
        st=time.time()
    PPinitBasis=np.matmul(np.linalg.inv(Pini),np.matmul(P,Pini))
    EPinitBasis=np.trace(np.dot(np.abs(PPinitBasis),np.log(np.abs(PPinitBasis))))
    with objmode(et='f8'):
         et=time.time()
    functioncalls[28]+=1
    timers['PgsEtimes']=np.append(timers['PgsEtimes'],et-st)
    return EPinitBasis,functioncalls,timers


def rttddft(nsteps,dt,propagator,SCFiterations,L,N_i,R_I,elements,basis_sets,basis_filename,kickstrength,kickdirection,projectname):
    with objmode(st='f8'):
        st=time.time()
    # Ground state stuff
    print(' *******           *******   ********** *******\n'+ 
    '/**////**  **   **/**////** /////**/// /**////**\n'+
    '/**   /** //** ** /**   /**     /**    /**   /**\n'+
    '/*******   //***  /*******      /**    /******* \n'+
    '/**////     /**   /**///**      /**    /**////  \n'+
    '/**         **    /**  //**     /**    /**      \n'+
    '/**        **     /**   //**    /**    /**      \n'+
    '//        //      //     //     //     //       \n'+
    '-------------------------------------------------------\n')
    print('Contributing authors:\nMatthew Thompson - MatThompson@lincoln.ac.uk\nMatt Watkins - MWatkins@lincoln.ac.uk\nWarren Lynch - WLynch@lincoln.ac.uk\n'+
    'Date of last edit: 18/07/2023\n'+
    'Description: A program to perform RT-TDDFT exclusively in Python.\n'+ 
    '\t     A variety of propagators (CN, EM, ETRS, etc.) can be \n'+
    '\t     selected for benchmarking and testing.\n'+
    '\n'+'System requirements:\n'+
    '- Python >= 3.10\n'+
    '- Numpy >= 1.13\n'+
    '- npy-append-array >= 0.9.5\n'+
    '- Sympy >=1.7.1\n'+
    '- Scipy >= 0.19\n'+
    '- PyLibXC\n'+
    '- Numba >= 0.51.2\n'+
    '--------------------------------------------------------------------\n')

    
    # Initialising the arrays containing the runtimes of each function

    timers=Dict.empty(
        key_type=types.unicode_type,
        value_type=float_array,
    )
    
    timers['GridCreatetimes']=np.array([])
    timers['constructGTOstimes']=np.array([])
    timers['constructCGFtimes']=np.array([])
    timers['basissettimes']=np.array([])
    timers['calculaterealspacedensitytimes']=np.array([])
    timers['calculatecoredensitytimes']=np.array([])
    timers['gridintegrationtimes']=np.array([])
    timers['energycalculationtimes']=np.array([])
    timers['calculateoverlaptimes']=np.array([])
    timers['calculatekineticderivativetimes']=np.array([])
    timers['calculatehartreereciprocaltimes']=np.array([])
    timers['calculatehartreerealtimes']=np.array([])
    timers['calculateXCtimes']=np.array([])
    timers['calculateVSRtimes']=np.array([])
    timers['calculateselfenergytimes']=np.array([])
    timers['calculateIItimes']=np.array([])
    timers['DFTsetuptimes']=np.array([])
    timers['computeE0times']=np.array([])
    timers['computeDFTtimes']=np.array([])
    timers['GaussianKicktimes']=np.array([])
    timers['propagatortimes']=np.array([])
    timers['propagatetimes']=np.array([])
    timers['tdttimes']=np.array([])
    timers['getptimes']=np.array([])
    timers['LagrangeExtrapolatetimes']=np.array([])
    timers['popanalysistimes']=np.array([])
    timers['SEtimes']=np.array([])
    timers['vNEtimes']=np.array([])
    timers['PgsEtimes']=np.array([])
    timers['P_initguesstimes']=np.array([])
    timers['PP_coefstimes']=np.array([])
    timers['RTTDDFTtimes']=np.array([])

    functioncalls=np.zeros((32,),dtype=int)

    # Performing calculation of all constant variables to remove the need to repeat it multiple times.
    Z_I,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi,Cpp,rPP,alpha_PP,functioncalls,timers=dftSetup(R_I,L,N_i,elements,basis_sets,basis_filename,functioncalls,timers)

    print('Ground state calculations:\n')
    # Compute the ground state first
    P,H,C,KS,functioncalls,timers=computeDFT_first(R_I,L,N_i,Z_I,Cpp,SCFiterations,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,S,delta_T,E_self,E_II,phi,rPP,alpha_PP,basis_filename,functioncalls,timers)
    # initialising all variable arrays
    Pgs=P
    energies=[]
    mu=[]
    propagationtimes=[]
    SE=[]
    vNE=[]
    EPinit=[]
    # Writing a short .txt file giving the simulation parameters to the project folder.
    # This will fail if the directory already exists, as intended, to prevent data loss.
    os.mkdir(os.path.join(os.getcwd(),projectname))
    lines = ['PyRTP run parameters','------------------------','Propagator: '+propagator,'Timesteps: '+str(nsteps),'Timestep: '+str(dt),'Kick strength: '+str(kickstrength),'Kick direction: '+str(kickdirection)]
    with open(projectname+'/'+projectname+'_parameters.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

    # time array can be set now
    t=np.arange(0,nsteps*dt,dt)

    for i in range(0,nsteps):
        print('--------------------------------------------------------------------\nPropagation timestep: '+str(i+1))
        #Applying perturbation
        KS,functioncalls,timers=GaussianKick(KS,kickstrength,kickdirection,t[i],r_x,r_y,r_z,phi,dr,functioncalls,timers)
        #Getting perturbed density matrix
        P,functioncalls,timers=GetP(KS,S,functioncalls,timers)
        # Propagating depending on method.
        # AETRS, CAETRS and CFM4 require 2 prior timesteps, which use ETRS
        if i<2 and propagator==('AETRS'):
            P,proptime,functioncalls,timers=propagate(R_I,Z_I,P,H,C,dt,'ETRS',N_i,Cpp,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i,phi,rPP,alpha_PP,functioncalls,timers)
        elif i<2 and propagator==('CAETRS'):
            P,proptime,functioncalls,timers=propagate(R_I,Z_I,P,H,C,dt,'ETRS',N_i,Cpp,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i,phi,rPP,alpha_PP,functioncalls,timers)
        elif i<2 and propagator==('CFM4'):
            P,proptime,functioncalls,timers=propagate(R_I,Z_I,P,H,C,dt,'ETRS',N_i,Cpp,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i,phi,rPP,alpha_PP,functioncalls,timers)
        elif i>1 and propagator==('AETRS'):
            P,proptime,functioncalls,timers=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,delta_T,E_self,E_II,L,energies[i-1],t,energies,t[i],i,phi,rPP,alpha_PP,functioncalls,timers)
        elif i>1 and propagator==('CAETRS'):
            P,proptime,functioncalls,timers=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,delta_T,E_self,E_II,L,energies[i-1],t,energies,t[i],i,phi,rPP,alpha_PP,functioncalls,timers)
        elif i>1 and propagator==('CFM4'):
            P,proptime,functioncalls,timers=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,delta_T,E_self,E_II,L,energies[i-1],t,energies,t[i],i,phi,rPP,alpha_PP,functioncalls,timers)
        else:
            P,proptime,functioncalls,timers=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i,phi,rPP,alpha_PP,functioncalls,timers)
        print('Propagation time: '+str(proptime))
        # Converging on accurate KS and P
        P,H,C,KS,functioncalls,timers=computeDFT(R_I,L,N_i,P,Z_I,Cpp,SCFiterations,r_x,r_y,r_z,N,dr,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi,rPP,alpha_PP,functioncalls,timers)
        # Information Collection
        D_x,D_y,D_z,D_tot,functioncalls,timers=transition_dipole_tensor_calculation(r_x,r_y,r_z,phi,dr,functioncalls,timers)
        mu_t=np.trace(np.dot(D_tot,P))
        energies.append(H)
        SEnow,functioncalls,timers=ShannonEntropy(P,functioncalls,timers)
        SE.append(SEnow)
        vNEnow,functioncalls,timers=vonNeumannEntropy(P,functioncalls,timers)
        vNE.append(vNEnow)
        PgsEnow,functioncalls,timers=EntropyPInitBasis(P,Pgs,functioncalls,timers)
        EPinit.append(PgsEnow)
        mu.append(mu_t)
        propagationtimes.append(proptime)
        # Saving data to external files every 10 steps
        if  i>0 and (i+1)%10==0:
            with NpyAppendArray(projectname+'/'+projectname+'_energies.npy') as npaa:
                npaa.append(np.array(energies[i-9:i+1]))
            with NpyAppendArray(projectname+'/'+projectname+'_mu.npy') as npaa:
                npaa.append(np.array(mu[i-9:i+1]))
            with NpyAppendArray(projectname+'/'+projectname+'_timings.npy') as npaa:
                npaa.append(np.array(propagationtimes[i-9:i+1]))
            with NpyAppendArray(projectname+'/'+projectname+'_SE.npy') as npaa:
                npaa.append(np.array(SE[i-9:i+1]))
            with NpyAppendArray(projectname+'/'+projectname+'_t.npy') as npaa:
                npaa.append(t[i-9:i+1])
            with NpyAppendArray(projectname+'/'+projectname+'_vNE.npy') as npaa:
                npaa.append(np.array(vNE[i-9:i+1]))
            with NpyAppendArray(projectname+'/'+projectname+'_EPinit.npy') as npaa:
                npaa.append(np.array(EPinit[i-9:i+1]))

        # Outputting calculated data
        print('Total dipole moment: '+str(mu_t))
        print('Shannon entropy: '+str(SE[i]))
        print('von Neumann Entropy: '+str(vNE[i]))
        print('P_init Basis Entropy: '+str(EPinit[i]))

    with objmode(et='f8'):
         et=time.time()
    functioncalls[31]+=1
    timers['RTTDDFTtimes']=np.append(timers['RTTDDFTtimes'],et-st)
    
    return t,energies,mu,propagationtimes,SE,vNE,EPinit,functioncalls,timers

#%%
# Simulation parameters
nsteps=2
timestep=0.1
SCFiterations=100
kickstrength=0.1
kickdirection=np.array([1,0,0])
proptype='CFM4'
projectname='CFM4run'

#Grid parameters
L=10.
N_i=100

# Molecule parameters
R_I = np.array([np.array([0.,0.,0.]), np.array([0.,0.,1.4632])]) #R_I[0] for He, R_I[1] for H.
elements = ['He','H']
basis_filename='BASIS_MOLOPT'
basis_sets=['SZV-MOLOPT-SR-GTH','SZV-MOLOPT-GTH']

t,energies,mu,timings,SE,vNE,EPinit,functioncalls,timers=rttddft(nsteps,timestep,proptype,SCFiterations,L,N_i,R_I,elements,basis_sets,basis_filename,kickstrength,kickdirection,projectname)

#%%
# Post-Processing section
#%%
# Energy plot
plt.plot(t,np.array(energies))
plt.xlabel('Time, $s$')
plt.ylabel('Energy, $Ha$')

#%%
# Dipole moment plot
plt.plot(t,np.array(mu))
plt.xlabel('Time, $s$')
plt.ylabel('Dipole moment, $\mu$')

#%% Padded dipole moment plot
mu_padded=np.append(np.array(mu),np.ones(2000-nsteps)*(np.mean(mu)))
t_padded=np.arange(0,2000*timestep,timestep)*2.418e-17
plt.plot(t_padded,mu_padded)

#%%
# Absorption Spectrum plot
filterpercentage=0
mu=np.array(mu)
c=299792458
h=6.62607015e-34
sp=scipy.fft.rfft(mu)
indexes=np.where(sp<np.percentile(sp,filterpercentage))[0]
sp[indexes]=0
freq = scipy.fft.rfftfreq(mu.size,(0.1*2.4188843265857e-17))
freqshift=scipy.fft.fftshift(freq)
ld=c/freq
en=h*freq
plt.plot(ld,np.abs(sp))
plt.xlabel('Wavelength, $\lambda$')
plt.ylabel('Intensity')
#plt.xlim([0, 5e-7])
#%%
# Padded Absorption Spectrum plot
filterpercentage=0
c=299792458
h=6.62607015e-34
sp=scipy.fft.rfft(mu_padded)
indexes=np.where(sp<np.percentile(sp,filterpercentage))[0]
sp[indexes]=0
freq = scipy.fft.rfftfreq(mu_padded.size,(0.1*2.4188843265857e-17))
freqshift=scipy.fft.fftshift(freq)
ld=c/freq
en=h*freq
plt.plot(ld,np.abs(sp))
plt.xlabel('Wavelength, $\lambda$')
plt.ylabel('Intensity')
#plt.xlim([0, 5e-7])
# %%
#Timings table
print(''.ljust(73,'-'))
print('|'+'Timings Table'.center(71)+'|')
print(''.ljust(73,'-'))
print('| '+'Function'.ljust(30)+'|'+'Function Calls'.center(19)+'|'+ 'Average run time'.center(19)+'|')
print(''.ljust(73,'-'))
print('| '+'GridCreate'.ljust(30)+'|'+str(functioncalls[0]).center(19)+'|'+ str(np.round(np.mean(GridCreatetimes),13)).center(19)+'|')
print('| '+'construct_GTOs'.ljust(30)+'|'+str(functioncalls[1]).center(19)+'|'+ str(np.round(np.mean(constructGTOstimes),13)).center(19)+'|')
print('| '+'construct_CGF'.ljust(30)+'|'+str(functioncalls[2]).center(19)+'|'+ str(np.round(np.mean(constructCGFtimes),13)).center(19)+'|')
#print('| '+'CP2K_basis_set_data'.ljust(30)+'|'+str(functioncalls[3]).center(19)+'|'+ str(np.round(np.mean(basissettimes),13)).center(19)+'|')
print('| '+'calculate_realspace_density'.ljust(30)+'|'+str(functioncalls[4]).center(19)+'|'+ str(np.round(np.mean(calculaterealspacedensitytimes),13)).center(19)+'|')
print('| '+'calculate_core_density'.ljust(30)+'|'+str(functioncalls[5]).center(19)+'|'+ str(np.round(np.mean(calculatecoredensitytimes),13)).center(19)+'|')
print('| '+'grid_integration'.ljust(30)+'|'+str(functioncalls[6]).center(19)+'|'+ str(np.round(np.mean(gridintegrationtimes),13)).center(19)+'|')
print('| '+'energy_calculation'.ljust(30)+'|'+str(functioncalls[7]).center(19)+'|'+ str(np.round(np.mean(energycalculationtimes),13)).center(19)+'|')
print('| '+'calculate_overlap'.ljust(30)+'|'+str(functioncalls[8]).center(19)+'|'+ str(np.round(np.mean(calculateoverlaptimes),13)).center(19)+'|')
print('| '+'calculate_kinetic_derivative'.ljust(30)+'|'+str(functioncalls[9]).center(19)+'|'+ str(np.round(np.mean(calculatekineticderivativetimes),13)).center(19)+'|')
print('| '+'calculate_hartree_reciprocal'.ljust(30)+'|'+str(functioncalls[10]).center(19)+'|'+ str(np.round(np.mean(calculatehartreereciprocaltimes),13)).center(19)+'|')
print('| '+'calculate_hartree_real'.ljust(30)+'|'+str(functioncalls[11]).center(19)+'|'+ str(np.round(np.mean(calculatehartreerealtimes),13)).center(19)+'|')
print('| '+'calculate_XC'.ljust(30)+'|'+str(functioncalls[12]).center(19)+'|'+ str(np.round(np.mean(calculateXCtimes),13)).center(19)+'|')
print('| '+'calculate_V_SR_r'.ljust(30)+'|'+str(functioncalls[13]).center(19)+'|'+ str(np.round(np.mean(calculateVSRtimes),13)).center(19)+'|')
print('| '+'calculate_self_energy'.ljust(30)+'|'+str(functioncalls[14]).center(19)+'|'+ str(np.round(np.mean(calculateselfenergytimes),13)).center(19)+'|')
print('| '+'calculate_Ion_interation'.ljust(30)+'|'+str(functioncalls[15]).center(19)+'|'+ str(np.round(np.mean(calculateIItimes),13)).center(19)+'|')
print('| '+'dftSetup'.ljust(30)+'|'+str(functioncalls[16]).center(19)+'|'+ str(np.round(np.mean(DFTsetuptimes),13)).center(19)+'|')
print('| '+'computeE_0'.ljust(30)+'|'+str(functioncalls[17]).center(19)+'|'+ str(np.round(np.mean(computeE0times),13)).center(19)+'|')
print('| '+'computeDFT'.ljust(30)+'|'+str(functioncalls[18]).center(19)+'|'+ str(np.round(np.mean(computeDFTtimes),13)).center(19)+'|')
print('| '+'GaussianKick'.ljust(30)+'|'+str(functioncalls[19]).center(19)+'|'+ str(np.round(np.mean(GaussianKicktimes),13)).center(19)+'|')
print('| '+'propagator'.ljust(30)+'|'+str(functioncalls[20]).center(19)+'|'+ str(np.round(np.mean(propagatortimes),13)).center(19)+'|')
print('| '+'propagate'.ljust(30)+'|'+str(functioncalls[21]).center(19)+'|'+ str(np.round(np.mean(propagatetimes),13)).center(19)+'|')
print('| '+'transition_dipole_tensor'.ljust(30)+'|'+str(functioncalls[22]).center(19)+'|'+ str(np.round(np.mean(tdttimes),13)).center(19)+'|')
print('| '+'GetP'.ljust(30)+'|'+str(functioncalls[23]).center(19)+'|'+ str(np.round(np.mean(getptimes),13)).center(19)+'|')
print('| '+'LagrangeExtrapolate'.ljust(30)+'|'+str(functioncalls[24]).center(19)+'|'+ str(np.round(np.mean(LagrangeExtrapolatetimes),13)).center(19)+'|')
#print('| '+'pop_analysis'.ljust(30)+'|'+str(functioncalls[25]).center(19)+'|'+ str(np.round(np.mean(popanalysistimes),13)).center(19)+'|')
print('| '+'ShannonEntropy'.ljust(30)+'|'+str(functioncalls[26]).center(19)+'|'+ str(np.round(np.mean(SEtimes),13)).center(19)+'|')
print('| '+'vonNeumannEntropy'.ljust(30)+'|'+str(functioncalls[27]).center(19)+'|'+ str(np.round(np.mean(vNEtimes),13)).center(19)+'|')
print('| '+'EntropyPInitBasis'.ljust(30)+'|'+str(functioncalls[28]).center(19)+'|'+ str(np.round(np.mean(PgsEtimes),13)).center(19)+'|')
print('| '+'P_init_guess_calc'.ljust(30)+'|'+str(functioncalls[29]).center(19)+'|'+ str(np.round(np.mean(P_initguesstimes),13)).center(19)+'|')
print('| '+'PP_coefs'.ljust(30)+'|'+str(functioncalls[30]).center(19)+'|'+ str(np.round(np.mean(PP_coefstimes),13)).center(19)+'|')
print('| '+'rttddft'.ljust(30)+'|'+str(functioncalls[31]).center(19)+'|'+ str(np.round(np.mean(RTTDDFTtimes),13)).center(19)+'|')
print(''.ljust(73,'-'))

# %%