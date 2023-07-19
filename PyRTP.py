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
from numba import jit,objmode # Translates Python script into machine code for significantly faster runtimes
import re # Used to read and format .txt and .dat files (reading basis set libraries, etc.)

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
def GridCreate(L,N_i,functioncalls,GridCreatetimes):
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
    GridCreatetimes=np.append(GridCreatetimes,et-st)
    return r_x,r_y,r_z,N,dr,functioncalls,GridCreatetimes

@jit(nopython=True,cache=True)
def GTO(a_GTO,r_GTO,R_GTO):
    # This function evaluates a Gaussian-type orbital evaluated on the grid from user defined coefficients and exponents, 
    # such that a_GTO = exponent, r_GTO = grid position vector and R_GTO = nuclei position vector. 
    GTOcalc=(2*a_GTO/np.pi)**(3/4)*np.exp(-a_GTO*(np.linalg.norm(r_GTO - R_GTO))**2)
    return GTOcalc

@jit(nopython=True,cache=True)
def construct_GTOs(nuc,N,N_i,r_x,r_y,r_z,R_I,Alpha,functioncalls,constructGTOstimes):
    # This function evaluates the Gaussian-type orbitals for a selected molecule at each point of the grid
    # and saves it to an array. In this case, as the orbital approximation being used is STO-3G, the array
    # has dimensions {3 x N_i x N_i x N_i}, as this approximation uses 3 Gaussian functions per orbital.

    with objmode(st='f8'):
        st=time.time()
    #create a matrix of grid points for each of the three GTOs, initialised to zero.
    GTO_p = np.zeros(len(Alpha)*N).reshape(len(Alpha),N_i,N_i,N_i)

    #Loop through GTOs and grid points, calculate the GTO value and assign to GTO_p.
    for gto in range(0,len(Alpha)) :
        #for i,j,k in range(0,N_i) :    
        for i in range(0,N_i) : 
            for j in range(0,N_i) :
                for k in range(0,N_i) :
                    p = np.array([r_x[i],r_y[j],r_z[k]]) #Select current grid position vector.

                    GTO_p[gto][i][j][k] = GTO(Alpha[gto],p,R_I) #calculate GTO value using GTO function call.
    with objmode(et='f8'):
         et=time.time()
    functioncalls[1]+=1
    constructGTOstimes=np.append(constructGTOstimes,et-st)
    return GTO_p,functioncalls,constructGTOstimes

@jit(nopython=True,cache=True)
def construct_CGF(GTOs,N,N_i,Coef,functioncalls,constructCGFtimes):
    # This function evaluates the STO-3G approximation at each grid point.
    with objmode(st='f8'):
        st=time.time()
    CGF = np.zeros(N).reshape(N_i,N_i,N_i) #create a matrix of grid points initialised to zero.
    for g in range(0,len(GTOs)): 
        CGF += Coef[g]*GTOs[g]              #construct the CGF from the GTOs and coefficients, Eq. 2.
    with objmode(et='f8'):
         et=time.time()
    functioncalls[2]+=1
    constructCGFtimes=np.append(constructCGFtimes,et-st)
    return CGF,functioncalls,constructCGFtimes

def CP2K_basis_set_data(filename,element,basis_set,functioncalls,basissettimes):
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
    if key[2] > 0. : coef = coef.reshape(int(key[2]),int(coef_range/key[2]),int(key[3])) #l,function,data                                                                                                                                            
    else : coef = coef.reshape(1,int(coef_range),int(key[3]))

    coef_setup=np.array(coef,dtype=float).reshape(len(alpha),1,1)
    coef_out=[]
    for i in range(0,len(alpha)):
        coef_out.append(coef_setup[i][0][0])
    
    coef=np.array(coef_out,dtype=float)
    alpha=np.array(alpha,dtype=float).reshape(len(coef))
    et=time.time()
    functioncalls[3]+=1
    basissettimes=np.append(basissettimes,et-st)
    return alpha,coef,functioncalls,basissettimes

@jit(nopython=True,cache=True)
def calculate_realspace_density(phi,N,N_i,P,dr,functioncalls,calculaterealspacedensitytimes) :
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
    calculaterealspacedensitytimes=np.append(calculaterealspacedensitytimes,et-st)
    return n_el_r, np.sum(n_el_r)*dr,functioncalls,calculaterealspacedensitytimes

@jit(nopython=True,cache=True)
def calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I,functioncalls,calculatecoredensitytimes) :
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
    calculatecoredensitytimes=np.append(calculatecoredensitytimes,et-st)
    return n_c_r,functioncalls,calculatecoredensitytimes

@jit(nopython=True,cache=True)
def grid_integration(V_r,dr,phi,functioncalls,gridintegrationtimes):
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
    gridintegrationtimes=np.append(gridintegrationtimes,et-st)
    return V,functioncalls,gridintegrationtimes

@jit(nopython=True,cache=True)
def energy_calculation(V,P,functioncalls,energycalculationtimes): 
    # This function evaluates a potential in the correct basis and determines the corresponding energy contribution. 
    with objmode(st='f8'):
        st=time.time()
    E=np.sum(P*V)
    with objmode(et='f8'):
         et=time.time()
    functioncalls[7]+=1
    energycalculationtimes=np.append(energycalculationtimes,et-st)
    return E,functioncalls,energycalculationtimes

@jit(nopython=True,cache=True)
def calculate_overlap(N,N_i,dr,phi,functioncalls,calculateoverlaptimes,gridintegrationtimes):
    # This function evaluates the overlap between atomic orbitals. It is worth noting that this function is simplified using
    # the grid integration function, as if the arbitrary potential being evaluated is equal to 1 at all points, the equations
    # to calculate the potential and the overlap are equivalent. Of course, this could be done explicitly in this function,
    # however it is more compact to do it this way.
    with objmode(st='f8'):
        st=time.time()
    V_temp = np.ones(N).reshape(N_i,N_i,N_i)
    S,functioncalls,gridintegrationtimes = grid_integration(V_temp,dr,phi,functioncalls,gridintegrationtimes)
    with objmode(et='f8'):
         et=time.time()
    functioncalls[8]+=1
    calculateoverlaptimes=np.append(calculateoverlaptimes,et-st)
    return S,functioncalls,calculateoverlaptimes,gridintegrationtimes

@jit(nopython=True,cache=True)
def calculate_kinetic_derivative(phi,phi_PW_G,N,N_i,G_u,G_v,G_w,L,functioncalls,calculatekineticderivativetimes) :
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
    calculatekineticderivativetimes=np.append(calculatekineticderivativetimes,et-st)
    return delta_T,functioncalls,calculatekineticderivativetimes

@jit(nopython=True,cache=True)
def calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L,functioncalls,calculatehartreereciprocaltimes):
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
         Vout=np.fft.ifftshift(Vg)
         et=time.time()
    functioncalls[10]+=1
    calculatehartreereciprocaltimes=np.append(calculatehartreereciprocaltimes,et-st)
    return Vout, E_hart_G,functioncalls,calculatehartreereciprocaltimes #result is shifted back. 

@jit(nopython=True,cache=True)
def calculate_hartree_real(V_r,n_r,dr,functioncalls,calculatehartreerealtimes) :
    with objmode(st='f8'):
        st=time.time()
    Har=0.5*np.sum(V_r*n_r)*dr
    with objmode(et='f8'):
         et=time.time()
    functioncalls[11]+=1
    calculatehartreerealtimes=np.append(calculatehartreerealtimes,et-st)
    return Har,functioncalls,calculatehartreerealtimes


def calculate_XC_pylibxc(n_el_r,N_i,dr,functioncalls,calculateXCtimes):
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
    calculateXCtimes=np.append(calculateXCtimes,et-st)
    return V_XC_r, E_XC,functioncalls,calculateXCtimes


@jit(nopython=True,cache=True)
def calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,alpha_PP,R_I,functioncalls,calculateVSRtimes) :
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
                        with objmode():
                            V_SR_r[i][j][k] += Cpp[n][c]*(np.sqrt(2.)*alpha_PP[n]*r)**(2*(c+1)-2)*np.exp(-(alpha_PP[n]*r)**2) 

    with objmode(et='f8'):
         et=time.time()
    functioncalls[13]+=1
    calculateVSRtimes=np.append(calculateVSRtimes,et-st)     
    return V_SR_r,functioncalls,calculateVSRtimes

@jit(nopython=True,cache=True)
def calculate_self_energy(Z_I,alpha_PP,functioncalls,calculateselfenergytimes) :
    # This function calculates the self energy correction term of the total energy, which is relatively simple.
    with objmode(st='f8'):
        st=time.time()
    R_pp = 1/alpha_PP #R_{I}^{c}
    self=np.sum(-(2*np.pi)**(-1/2)*Z_I**2/R_pp)
    with objmode(et='f8'):
         et=time.time()
    functioncalls[14]+=1
    calculateselfenergytimes=np.append(calculateselfenergytimes,et-st)
    return self,functioncalls,calculateselfenergytimes

@jit(nopython=True,cache=True)
def calculate_Ion_interaction(Z_I,R_I,alpha_PP,functioncalls,calculateIItimes) :
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
    calculateIItimes=np.append(calculateIItimes,et-st)
    return E_II,functioncalls,calculateIItimes

def P_init_guess_calc(T,S,N,N_i,Z_I,r_x,r_y,r_z,R_I,functioncalls,calculatecoredensitytimes,G_u,G_v,G_w,L,calculatehartreereciprocaltimes,dr,phi,gridintegrationtimes,P_initguesstimes):
    # This function uses exclusively the core Hamiltonian to approximate the initial density matrix. This is then saved in the src/InitialGuesses directory
    # as a .npy file. When calling this function again, the program will check this directory for pre-calculated initial guesses, and if none are found, 
    # the function will compute it. It is worth noting that using the core Hamiltonian and the Hartree-Fock method is only suitable for small molecules.
    with objmode(st='f8'):
        st=time.time()
    Z_Iparts=Z_I.flatten().tolist()
    R_Iparts=R_I.flatten().tolist()
    Z_Istring = "".join([str(i) for i in Z_Iparts])
    R_Istring = "".join([str(i) for i in R_Iparts])
    
    filenamelist=['STO-3G',Z_Istring,R_Istring]
    filename="".join(filenamelist)
    if os.path.isfile("src/InitialGuesses/"+filename+".npy"):
        print("Reading " +filename+".npy from src/InitialGuesses")
        P_init = np.load("src/InitialGuesses/"+filename+".npy")
    else:
        n_c_r,functioncalls,calculatecoredensitytimes = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I,functioncalls,calculatecoredensitytimes)
        n_c_G=np.fft.fftn(n_c_r)
        V_c_G, E_c_G,functioncalls,calculatehartreereciprocaltimes = calculate_hartree_reciprocal(n_c_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L,functioncalls,calculatehartreereciprocaltimes)
        V_c_r = np.fft.ifftn(V_c_G)
        V_nuc,functioncalls,gridintegrationtimes = grid_integration(V_c_r,dr,phi,functioncalls,gridintegrationtimes)
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
    P_initguesstimes=np.append(P_initguesstimes,et-st)
    return P_init,functioncalls,calculatecoredensitytimes,calculatehartreereciprocaltimes,gridintegrationtimes,P_initguesstimes

def PP_coefs(Z_I,functioncalls,PP_coefstimes):
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
    PP_coefstimes=np.append(PP_coefstimes,et-st)
    return np.array(cPP).astype(np.float64),np.array(rPP).astype(np.float64),alpha_PP.astype(np.float64),functioncalls,PP_coefstimes

def dftSetup(R_I,alpha,Coef,L,N_i,Z_I,functioncalls,GridCreatetimes,constructGTOstimes,constructCGFtimes,calculateoverlaptimes,gridintegrationtimes,calculatekineticderivativetimes,calculateselfenergytimes,calculateIItimes,DFTsetuptimes,basissettimes,PP_coefstimes):
    # This function calls all functions required to perform DFT that do not change during the SCF cycle.
    # This prevents the functions being called unneccessarily. This function should be called prior to the 
    # 'computeDFT' function.

    with objmode(st='f8'):
        st=time.time()
    r_x,r_y,r_z,N,dr,functioncalls,GridCreatetimes=GridCreate(L,N_i,functioncalls,GridCreatetimes)
    #alpha_H, coef_H,functioncalls,basissettimes = CP2K_basis_set_data('BASIS_MOLOPT','H','SZV-MOLOPT-GTH',functioncalls,basissettimes)
    #alpha_He, coef_He,functioncalls,basissettimes = CP2K_basis_set_data('BASIS_MOLOPT','He','SZV-MOLOPT-SR-GTH',functioncalls,basissettimes)
    GTOs_He,functioncalls,constructGTOstimes = construct_GTOs('He',N,N_i,r_x,r_y,r_z,R_I[0],alpha[0],functioncalls,constructGTOstimes)
    CGF_He,functioncalls,constructCGFtimes = construct_CGF(GTOs_He,N,N_i,Coef,functioncalls,constructCGFtimes)
    GTOs_H,functioncalls,constructGTOstimes = construct_GTOs('H',N,N_i,r_x,r_y,r_z,R_I[1],alpha[1],functioncalls,constructGTOstimes)
    CGF_H,functioncalls,constructCGFtimes = construct_CGF(GTOs_H,N,N_i,Coef,functioncalls,constructCGFtimes)
    phi = np.array([CGF_He,CGF_H])
    G_u = np.linspace(-N_i*np.pi/L,N_i*np.pi/L,N_i,endpoint=False)
    G_v = np.linspace(-N_i*np.pi/L,N_i*np.pi/L,N_i,endpoint=False)
    G_w = np.linspace(-N_i*np.pi/L,N_i*np.pi/L,N_i,endpoint=False)
    PW_He_G = np.fft.fftshift(np.fft.fftn(CGF_He))
    PW_H_G = np.fft.fftshift(np.fft.fftn(CGF_H))
    phi_PW_G = np.array([PW_He_G,PW_H_G])
    S, functioncalls, calculateoverlaptimes, gridintegrationtimes = calculate_overlap(N,N_i,dr,phi,functioncalls,calculateoverlaptimes,gridintegrationtimes)
    delta_T,functioncalls,calculatekineticderivativetimes = calculate_kinetic_derivative(phi,phi_PW_G,N,N_i,G_u,G_v,G_w,L,functioncalls,calculatekineticderivativetimes)
    cPP,rPP,alpha_PP,functioncalls,PP_coefstimes=PP_coefs(Z_I,functioncalls,PP_coefstimes)
    print(cPP)
    E_self,functioncalls,calculateselfenergytimes = calculate_self_energy(Z_I,alpha_PP,functioncalls,calculateselfenergytimes)
    E_II,functioncalls,calculateIItimes = calculate_Ion_interaction(Z_I,R_I,alpha_PP,functioncalls,calculateIItimes)
    with objmode(et='f8'):
         et=time.time()
    functioncalls[16]+=1
    DFTsetuptimes=np.append(DFTsetuptimes,et-st)
    return r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi,cPP,rPP,alpha_PP,functioncalls,GridCreatetimes,constructGTOstimes,constructCGFtimes,calculateoverlaptimes,gridintegrationtimes,calculatekineticderivativetimes,calculateselfenergytimes,calculateIItimes,DFTsetuptimes,basissettimes,PP_coefstimes


def computeE_0(R_I,Z_I,P,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,rPP,alpha_PP,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times):
    # This function returns the total energy for a given density matrix P.
    with objmode(st='f8'):
        st=time.time()
    n_el_r, n_el_r_tot,functioncalls,calculaterealspacedensitytimes  = calculate_realspace_density(phi,N,N_i,P,dr,functioncalls,calculaterealspacedensitytimes)
    n_c_r,functioncalls,calculatecoredensitytimes = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I,functioncalls,calculatecoredensitytimes)
    n_r = n_el_r + n_c_r
    T,functioncalls,energycalculationtimes = energy_calculation(delta_T,P,functioncalls,energycalculationtimes)
    n_G = np.fft.fftn(n_r)
    V_G, E_hart_G,functioncalls,calculatehartreereciprocaltimes = calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L,functioncalls,calculatehartreereciprocaltimes)
    V_r = np.fft.ifftn(V_G)
    V_hart,functioncalls,gridintegrationtimes = grid_integration(V_r,dr,phi,functioncalls,gridintegrationtimes)
    E_hart_r,functioncalls,calculatehartreerealtimes = calculate_hartree_real(V_r,n_r,dr,functioncalls,calculatehartreerealtimes)
    V_XC_r,E_XC,functioncalls,calculateXCtimes = calculate_XC_pylibxc(n_el_r,N_i,dr,functioncalls,calculateXCtimes)
    V_XC,functioncalls,gridintegrationtimes = grid_integration(V_XC_r,dr,phi,functioncalls,gridintegrationtimes)
    V_SR_r,functioncalls,calculateVSRtimes = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,alpha_PP,R_I,functioncalls,calculateVSRtimes)
    V_SR,functioncalls,gridintegrationtimes = grid_integration(V_SR_r,dr,phi,functioncalls,gridintegrationtimes)
    E_SR,functioncalls,energycalculationtimes = energy_calculation(V_SR,P,functioncalls,energycalculationtimes)
    E_0 = E_hart_r + E_XC + E_SR + T + E_self + E_II
    with objmode(et='f8'):
         et=time.time()
    functioncalls[17]+=1
    computeE0times=np.append(computeE0times,et-st)
    return E_0,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times

def computeDFT_first(R_I,alpha,Coef,L,N_i,Z_I,Cpp,iterations,r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi,rPP,alpha_PP,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeDFTtimes,P_initguesstimes):
    # This function performs conventional ground-state DFT to produce a converged ground state electron density, wavefunction, energy and Kohn-Sham orbitals.
    # The 'DFTsetup' function must be called prior to calling this function.
    with objmode(st='f8'):
        st=time.time()
    P_init,functioncalls,calculatecoredensitytimes,calculatehartreereciprocaltimes,gridintegrationtimes,P_initguesstimes = P_init_guess_calc(delta_T,S,N,N_i,Z_I,r_x,r_y,r_z,R_I,functioncalls,calculatecoredensitytimes,G_u,G_v,G_w,L,calculatehartreereciprocaltimes,dr,phi,gridintegrationtimes,P_initguesstimes)
    P=P_init
    n_el_r, n_el_r_tot,functioncalls,calculaterealspacedensitytimes  = calculate_realspace_density(phi,N,N_i,P,dr,functioncalls,calculaterealspacedensitytimes)
    n_c_r,functioncalls,calculatecoredensitytimes = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I,functioncalls,calculatecoredensitytimes)
    n_r = n_el_r + n_c_r
    T,functioncalls,energycalculationtimes = energy_calculation(delta_T,P,functioncalls,energycalculationtimes)
    n_G = np.fft.fftn(n_r)
    V_G, E_hart_G,functioncalls,calculatehartreereciprocaltimes = calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L,functioncalls,calculatehartreereciprocaltimes)
    V_r = np.fft.ifftn(V_G)
    V_hart,functioncalls,gridintegrationtimes = grid_integration(V_r,dr,phi,functioncalls,gridintegrationtimes)
    E_hart_r,functioncalls,calculatehartreerealtimes = calculate_hartree_real(V_r,n_r,dr,functioncalls,calculatehartreerealtimes)
    V_XC_r,E_XC,functioncalls,calculateXCtimes = calculate_XC_pylibxc(n_el_r,N_i,dr,functioncalls,calculateXCtimes)
    V_XC,functioncalls,gridintegrationtimes = grid_integration(V_XC_r,dr,phi,functioncalls,gridintegrationtimes)
    V_SR_r,functioncalls,calculateVSRtimes = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,alpha_PP,R_I,functioncalls,calculateVSRtimes)
    V_SR,functioncalls,gridintegrationtimes = grid_integration(V_SR_r,dr,phi,functioncalls,gridintegrationtimes)
    E_SR,functioncalls,energycalculationtimes = energy_calculation(V_SR,P,functioncalls,energycalculationtimes)
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
        n_el_r, n_el_r_tot,functioncalls,calculaterealspacedensitytimes  = calculate_realspace_density(phi,N,N_i,P,dr,functioncalls,calculaterealspacedensitytimes)
        n_c_r,functioncalls,calculatecoredensitytimes = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I,functioncalls,calculatecoredensitytimes)
        n_r = n_el_r + n_c_r
        n_G = np.fft.fftn(n_r)
        V_G, E_hart_G,functioncalls,calculatehartreereciprocaltimes = calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L,functioncalls,calculatehartreereciprocaltimes)
        V_r = np.fft.ifftn(V_G)
        V_hart,functioncalls,gridintegrationtimes = grid_integration(V_r,dr,phi,functioncalls,gridintegrationtimes)
        E_hart_r,functioncalls,calculatehartreerealtimes = calculate_hartree_real(V_r,n_r,dr,functioncalls,calculatehartreerealtimes)
        V_XC_r,E_XC,functioncalls,calculateXCtimes = calculate_XC_pylibxc(n_el_r,N_i,dr,functioncalls,calculateXCtimes)
        V_XC,functioncalls,gridintegrationtimes = grid_integration(V_XC_r,dr,phi,functioncalls,gridintegrationtimes)
        V_SR_r,functioncalls,calculateVSRtimes = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,alpha_PP,R_I,functioncalls,calculateVSRtimes)
        V_SR,functioncalls,gridintegrationtimes = grid_integration(V_SR_r,dr,phi,functioncalls,gridintegrationtimes)
        E_SR,functioncalls,energycalculationtimes = energy_calculation(V_SR,P,functioncalls,energycalculationtimes)
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
    computeDFTtimes=np.append(computeDFTtimes,et-st)
    return P,np.real(E_0),C,KS,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeDFTtimes,P_initguesstimes

def computeDFT(R_I,alpha,Coef,L,N_i,P_init,Z_I,Cpp,iterations,r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi,rPP,alpha_PP,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeDFTtimes):
    # This function performs conventional ground-state DFT to produce a converged ground state electron density, wavefunction, energy and Kohn-Sham orbitals.
    # The 'DFTsetup' function must be called prior to calling this function.
    with objmode(st='f8'):
        st=time.time()
    P = P_init
    n_el_r, n_el_r_tot,functioncalls,calculaterealspacedensitytimes  = calculate_realspace_density(phi,N,N_i,P,dr,functioncalls,calculaterealspacedensitytimes)
    n_c_r,functioncalls,calculatecoredensitytimes = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I,functioncalls,calculatecoredensitytimes)
    n_r = n_el_r + n_c_r
    T,functioncalls,energycalculationtimes = energy_calculation(delta_T,P,functioncalls,energycalculationtimes)
    n_G = np.fft.fftn(n_r)
    V_G, E_hart_G,functioncalls,calculatehartreereciprocaltimes = calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L,functioncalls,calculatehartreereciprocaltimes)
    V_r = np.fft.ifftn(V_G)
    V_hart,functioncalls,gridintegrationtimes = grid_integration(V_r,dr,phi,functioncalls,gridintegrationtimes)
    E_hart_r,functioncalls,calculatehartreerealtimes = calculate_hartree_real(V_r,n_r,dr,functioncalls,calculatehartreerealtimes)
    V_XC_r,E_XC,functioncalls,calculateXCtimes = calculate_XC_pylibxc(n_el_r,N_i,dr,functioncalls,calculateXCtimes)
    V_XC,functioncalls,gridintegrationtimes = grid_integration(V_XC_r,dr,phi,functioncalls,gridintegrationtimes)
    V_SR_r,functioncalls,calculateVSRtimes = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,alpha_PP,R_I,functioncalls,calculateVSRtimes)
    V_SR,functioncalls,gridintegrationtimes = grid_integration(V_SR_r,dr,phi,functioncalls,gridintegrationtimes)
    E_SR,functioncalls,energycalculationtimes = energy_calculation(V_SR,P,functioncalls,energycalculationtimes)
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
        n_el_r, n_el_r_tot,functioncalls,calculaterealspacedensitytimes  = calculate_realspace_density(phi,N,N_i,P,dr,functioncalls,calculaterealspacedensitytimes)
        n_c_r,functioncalls,calculatecoredensitytimes = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I,functioncalls,calculatecoredensitytimes)
        n_r = n_el_r + n_c_r
        n_G = np.fft.fftn(n_r)
        V_G, E_hart_G,functioncalls,calculatehartreereciprocaltimes = calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L,functioncalls,calculatehartreereciprocaltimes)
        V_r = np.fft.ifftn(V_G)
        V_hart,functioncalls,gridintegrationtimes = grid_integration(V_r,dr,phi,functioncalls,gridintegrationtimes)
        E_hart_r,functioncalls,calculatehartreerealtimes = calculate_hartree_real(V_r,n_r,dr,functioncalls,calculatehartreerealtimes)
        V_XC_r,E_XC,functioncalls,calculateXCtimes = calculate_XC_pylibxc(n_el_r,N_i,dr,functioncalls,calculateXCtimes)
        V_XC,functioncalls,gridintegrationtimes = grid_integration(V_XC_r,dr,phi,functioncalls,gridintegrationtimes)
        V_SR_r,functioncalls,calculateVSRtimes = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,alpha_PP,R_I,functioncalls,calculateVSRtimes)
        V_SR,functioncalls,gridintegrationtimes = grid_integration(V_SR_r,dr,phi,functioncalls,gridintegrationtimes)
        E_SR,functioncalls,energycalculationtimes = energy_calculation(V_SR,P,functioncalls,energycalculationtimes)
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
    computeDFTtimes=np.append(computeDFTtimes,et-st)
    return P,np.real(E_0),C,KS,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeDFTtimes

# GaussianKick - provides energy to the system in the form of an electric pulse
# with a Gaussian envelop (run each propagation step!)
def GaussianKick(KS,scale,direction,t,r_x,r_y,r_z,phi,dr,functioncalls,GaussianKicktimes,tdttimes,gridintegrationtimes):
    with objmode(st='f8'):
        st=time.time()

    t0=1 # These parameters, representing the pulse center and width
    w=0.2 # were selected to be ideal for most systems.
             # In future, it is possible that the user could be given some
             # control over these parameters.

    Efield=np.dot(scale*np.exp((-(t-t0)**2)/(2*(w**2))),direction)
    D_x,D_y,D_z,D_tot,functioncalls,tdttimes,gridintegrationtimes=transition_dipole_tensor_calculation(r_x,r_y,r_z,phi,dr,functioncalls,tdttimes,gridintegrationtimes)
    V_app=-(np.dot(D_x,Efield[0])+np.dot(D_y,Efield[1])+np.dot(D_z,Efield[2]))
    KS_new=KS+V_app
    with objmode(et='f8'):
         et=time.time()
    functioncalls[19]+=1
    GaussianKicktimes=np.append(GaussianKicktimes,et-st)
    return KS_new,functioncalls,GaussianKicktimes,tdttimes,gridintegrationtimes

# Propagator - this function selects the equation to determine the unitary operator to propagate 
# the system forward in time a singular timestep, using predictor-corrector regime (if applicable)
# This function uses a match-case statement, hence the need for Python 3.10 or greater.
def propagator(select,H1,H2,dt,functioncalls,propagatortimes): #propagator functions
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
    propagatortimes=np.append(propagatortimes,et-st)
    return U,functioncalls,propagatortimes

def propagate(R_I,Z_I,P,H,C,dt,select,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,Hprev,t,energies,tnew,i,phi,rPP,alpha_PP,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times):
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
                H_p,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[0:i],energies[0:i],t[i-1]+(1/2),functioncalls,LagrangeExtrapolatetimes)  # LagrangeExtrapolate fits a curve to the previous set of energies
                                                                                                                                                          # and extrapolates based on that. It is important that the number of
            else:                                                                                                                                         # points be limited so as not to cause instability.
                H_p,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],t[i-1]+(1/2),functioncalls,LagrangeExtrapolatetimes)  

            U,functioncalls,propagatortimes=propagator(select,H_p,[],dt,functioncalls,propagatortimes) # calling the propagtor function to determine the unitary operator U.
            U=np.real(U)
            C_p=np.dot(U,C)
            # This is the updating section
            P_p=np.zeros_like(P)
            for u in range(0,len(P_p)):
                for v in range(0,len(P_p)) :
                    for p in range(0,len(P_p)-1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,rPP,alpha_PP,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
            # This is the correcting section
            H_c=H+(1/2)*(E_0-H)
            U,functioncalls,propagatortimes=propagator(select,H_c,[],dt,functioncalls,propagatortimes) # After the predictor-corrector regime, the unitary operator U is likely more accurate.
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
                H_p,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[0:i],energies[0:i],t[i-1]+(1/2),functioncalls,LagrangeExtrapolatetimes)  
            else:
                H_p,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],t[i-1]+(1/2),functioncalls,LagrangeExtrapolatetimes)    
            U,functioncalls,propagatortimes=propagator(select,H_p,[],dt,functioncalls,propagatortimes)
            U=np.real(U)
            C_p=np.dot(U,C)
            # This is the updating section
            P_p=np.zeros_like(P)
            for u in range(0,len(P_p)):
                for v in range(0,len(P_p)) :
                    for p in range(0,len(P_p)-1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,rPP,alpha_PP,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
            # This is the correcting section
            H_c=H+(1/2)*(E_0-H)
            U,functioncalls,propagatortimes=propagator(select,H_c,[],dt,functioncalls,propagatortimes)
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
                H_p,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[0:i],energies[0:i],t[i-1]+(1/2),functioncalls,LagrangeExtrapolatetimes) 
            else:
                H_p,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],t[i-1]+(1/2),functioncalls,LagrangeExtrapolatetimes)     
            U,functioncalls,propagatortimes=propagator('EM',H_p,[],dt,functioncalls,propagatortimes)
            U=np.real(U)
            C_p=np.dot(U,C)
            # This is the updating section of the EM propagator
            P_p=np.zeros_like(P)
            for u in range(0,len(P_p)):
                for v in range(0,len(P_p)) :
                    for p in range(0,len(P_p)-1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,rPP,alpha_PP,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
            # This is the correcting section of the EM propagator
            H_c=H+(1/2)*(E_0-H)
            U,functioncalls,propagatortimes=propagator('EM',H_c,[],dt,functioncalls,propagatortimes)
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
            Hdt,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=computeE_0(R_I,Z_I,P_new,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,rPP,alpha_PP,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
            # The ETRS propagator can now be called
            U,functioncalls,propagatortimes=propagator(select,H,Hdt,dt,functioncalls,propagatortimes)
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
                Hdt,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[0:i],energies[0:i],tnew,functioncalls,LagrangeExtrapolatetimes) 
            else:
                Hdt,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],tnew,functioncalls,LagrangeExtrapolatetimes)   
            U,functioncalls,propagatortimes=propagator(select,H,Hdt,dt,functioncalls,propagatortimes)
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
                Hdt,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[0:i],energies[0:i],tnew,functioncalls,LagrangeExtrapolatetimes) 
            else:
                Hdt,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],tnew,functioncalls,LagrangeExtrapolatetimes) 
            # This is the updating section
            U,functioncalls,propagatortimes=propagator(select,H,Hdt,dt,functioncalls,propagatortimes)
            U=np.real(U)
            C_p=np.dot(U,C)
            P_p=np.zeros_like(P)
            for u in range(0,len(P_p)):
                for v in range(0,len(P_p)) :
                    for p in range(0,len(P_p)-1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,rPP,alpha_PP,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
            # This is the correcting section
            H_c=H+(1/2)*(E_0-H)
            Hdt=2*H_c-Hprev
            U,functioncalls,propagatortimes=propagator(select,H_c,Hdt,dt,functioncalls,propagatortimes)
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
                Ht1,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[0:i],energies[0:i],(t[i-1])+((1/2)-(np.sqrt(3)/6)),functioncalls,LagrangeExtrapolatetimes)
                Ht2,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[0:i],energies[0:i],(t[i-1])+((1/2)+(np.sqrt(3)/6)),functioncalls,LagrangeExtrapolatetimes) 
            else:
                Ht1,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],(t[i-1])+((1/2)-(np.sqrt(3)/6)),functioncalls,LagrangeExtrapolatetimes)
                Ht2,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],(t[i-1])+((1/2)+(np.sqrt(3)/6)),functioncalls,LagrangeExtrapolatetimes) 
            
            U,functioncalls,propagatortimes=propagator(select,Ht1,Ht2,dt,functioncalls,propagatortimes)
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
    propagatetimes=np.append(propagatetimes,et-st)
    return P_new,et-st,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times

@jit(nopython=True,cache=True)
def transition_dipole_tensor_calculation(r_x,r_y,r_z,phi,dr,functioncalls,tdttimes,gridintegrationtimes) :
    # This function determines the transition dipole tensor in the x, y and z directions.
    # This is used for calculation of the Gaussian pulse and the dipole moment.
    with objmode(st='f8'):
        st=time.time()
    D_x,functioncalls,gridintegrationtimes = grid_integration(r_x,dr,phi,functioncalls,gridintegrationtimes)             
    D_y,functioncalls,gridintegrationtimes = grid_integration(r_y,dr,phi,functioncalls,gridintegrationtimes)
    D_z,functioncalls,gridintegrationtimes = grid_integration(r_z,dr,phi,functioncalls,gridintegrationtimes)   

    D_tot=D_x+D_y+D_z
    with objmode(et='f8'):
         et=time.time()
    functioncalls[22]+=1
    tdttimes=np.append(tdttimes,et-st)
    return D_x,D_y,D_z,D_tot,functioncalls,tdttimes,gridintegrationtimes

def GetP(KS,S,functioncalls,getptimes):
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
    getptimes=np.append(getptimes,et-st)
    return P_new,functioncalls,getptimes

def LagrangeExtrapolate(t,H,tnew,functioncalls,LagrangeExtrapolatetimes):
    # This function performs Lagrange extrapolation (see Scipy documentation for details).
    with objmode(st='f8'):
        st=time.time()
    f=lagrange(t,H)
    val=np.real(f(tnew))
    with objmode(et='f8'):
         et=time.time()
    functioncalls[24]+=1
    LagrangeExtrapolatetimes=np.append(LagrangeExtrapolatetimes,et-st)
    return val,functioncalls,LagrangeExtrapolatetimes

@jit(nopython=True,cache=True)
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

def ShannonEntropy(P,functioncalls,SEtimes):
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
    SEtimes=np.append(SEtimes,et-st)
    return SE,functioncalls,SEtimes

def vonNeumannEntropy(P,functioncalls,vNEtimes):
    # This function determines the von Neumann entropy, which represents the 'distance' from a pure state in this program.
    with objmode(st='f8'):
        st=time.time()
    vNE=np.trace(np.dot(P,np.log(P)))
    with objmode(et='f8'):
         et=time.time()
    functioncalls[27]+=1
    vNEtimes=np.append(vNEtimes,et-st)
    return vNE,functioncalls,vNEtimes

def EntropyPInitBasis(P,Pini,functioncalls,PgsEtimes):
    # This function produces an entropy-like measure based on the initial density matrix.
    with objmode(st='f8'):
        st=time.time()
    PPinitBasis=np.matmul(np.linalg.inv(Pini),np.matmul(P,Pini))
    EPinitBasis=np.trace(np.dot(np.abs(PPinitBasis),np.log(np.abs(PPinitBasis))))
    with objmode(et='f8'):
         et=time.time()
    functioncalls[28]+=1
    PgsEtimes=np.append(PgsEtimes,et-st)
    return EPinitBasis,functioncalls,PgsEtimes


def rttddft(nsteps,dt,propagator,SCFiterations,L,N_i,alpha,Coef,R_I,Z_I,kickstrength,kickdirection,projectname):
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

    print('Ground state calculations:\n')
    # Initialising the arrays containing the runtimes of each function
    GridCreatetimes=np.array([])
    constructGTOstimes=np.array([])
    constructCGFtimes=np.array([])
    basissettimes=np.array([])
    calculaterealspacedensitytimes=np.array([])
    calculatecoredensitytimes=np.array([])
    gridintegrationtimes=np.array([])
    energycalculationtimes=np.array([])
    calculateoverlaptimes=np.array([])
    calculatekineticderivativetimes=np.array([])
    calculatehartreereciprocaltimes=np.array([])
    calculatehartreerealtimes=np.array([])
    calculateXCtimes=np.array([])
    calculateVSRtimes=np.array([])
    calculateselfenergytimes=np.array([])
    calculateIItimes=np.array([])
    DFTsetuptimes=np.array([])
    computeE0times=np.array([])
    computeDFTtimes=np.array([])
    GaussianKicktimes=np.array([])
    propagatortimes=np.array([])
    propagatetimes=np.array([])
    tdttimes=np.array([])
    getptimes=np.array([])
    LagrangeExtrapolatetimes=np.array([])
    popanalysistimes=np.array([])
    SEtimes=np.array([])
    vNEtimes=np.array([])
    PgsEtimes=np.array([])
    P_initguesstimes=np.array([])
    PP_coefstimes=np.array([])
    RTTDDFTtimes=np.array([])
    functioncalls=np.zeros((32,),dtype=int)

    # Performing calculation of all constant variables to remove the need to repeat it multiple times.
    r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi,Cpp,rPP,alpha_PP,functioncalls,GridCreatetimes,constructGTOstimes,constructCGFtimes,calculateoverlaptimes,gridintegrationtimes,calculatekineticderivativetimes,calculateselfenergytimes,calculateIItimes,DFTsetuptimes,basissettimes,PP_coefstimes=dftSetup(R_I,alpha,Coef,L,N_i,Z_I,functioncalls,GridCreatetimes,constructGTOstimes,constructCGFtimes,calculateoverlaptimes,gridintegrationtimes,calculatekineticderivativetimes,calculateselfenergytimes,calculateIItimes,DFTsetuptimes,basissettimes,PP_coefstimes)

    # Compute the ground state first
    P,H,C,KS,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeDFTtimes,P_initguesstimes=computeDFT_first(R_I,alpha,Coef,L,N_i,Z_I,Cpp,SCFiterations,r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi,rPP,alpha_PP,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeDFTtimes,P_initguesstimes)
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
        KS,functioncalls,GaussianKicktimes,tdttimes,gridintegrationtimes=GaussianKick(KS,kickstrength,kickdirection,t[i],r_x,r_y,r_z,phi,dr,functioncalls,GaussianKicktimes,tdttimes,gridintegrationtimes)
        #Getting perturbed density matrix
        P,functioncalls,getptimes=GetP(KS,S,functioncalls,getptimes)
        # Propagating depending on method.
        # AETRS, CAETRS and CFM4 require 2 prior timesteps, which use ETRS
        if i<2 and propagator==('AETRS'):
            P,proptime,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=propagate(R_I,Z_I,P,H,C,dt,'ETRS',N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i,phi,rPP,alpha_PP,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
        elif i<2 and propagator==('CAETRS'):
            P,proptime,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=propagate(R_I,Z_I,P,H,C,dt,'ETRS',N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i,phi,rPP,alpha_PP,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
        elif i<2 and propagator==('CFM4'):
            P,proptime,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=propagate(R_I,Z_I,P,H,C,dt,'ETRS',N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i,phi,rPP,alpha_PP,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
        elif i>1 and propagator==('AETRS'):
            P,proptime,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,energies[i-1],t,energies,t[i],i,phi,rPP,alpha_PP,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
        elif i>1 and propagator==('CAETRS'):
            P,proptime,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,energies[i-1],t,energies,t[i],i,phi,rPP,alpha_PP,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
        elif i>1 and propagator==('CFM4'):
            P,proptime,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,energies[i-1],t,energies,t[i],i,phi,rPP,alpha_PP,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
        else:
            P,proptime,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i,phi,rPP,alpha_PP,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
        print('Propagation time: '+str(proptime))
        # Converging on accurate KS and P
        P,H,C,KS,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeDFTtimes=computeDFT(R_I,alpha,Coef,L,N_i,P,Z_I,Cpp,SCFiterations,r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi,rPP,alpha_PP,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeDFTtimes)
        # Information Collection
        D_x,D_y,D_z,D_tot,functioncalls,tdttimes,gridintegrationtimes=transition_dipole_tensor_calculation(r_x,r_y,r_z,phi,dr,functioncalls,tdttimes,gridintegrationtimes)
        mu_t=np.trace(np.dot(D_tot,P))
        energies.append(H)
        SEnow,functioncalls,SEtimes=ShannonEntropy(P,functioncalls,SEtimes)
        SE.append(SEnow)
        vNEnow,functioncalls,vNEtimes=vonNeumannEntropy(P,functioncalls,vNEtimes)
        vNE.append(vNEnow)
        PgsEnow,functioncalls,PgsEtimes=EntropyPInitBasis(P,Pgs,functioncalls,PgsEtimes)
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
    RTTDDFTtimes=np.append(RTTDDFTtimes,et-st)
    
    return t,energies,mu,propagationtimes,SE,vNE,EPinit,functioncalls,GridCreatetimes,constructGTOstimes,constructCGFtimes,calculaterealspacedensitytimes,calculatecoredensitytimes,gridintegrationtimes,energycalculationtimes,calculateoverlaptimes,calculatekineticderivativetimes,calculatehartreereciprocaltimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,calculateselfenergytimes,calculateIItimes,DFTsetuptimes,computeE0times,computeDFTtimes,GaussianKicktimes,propagatortimes,propagatetimes,tdttimes,getptimes,LagrangeExtrapolatetimes,popanalysistimes,SEtimes,vNEtimes,PgsEtimes,RTTDDFTtimes,P_initguesstimes,PP_coefstimes

#%%
# Simulation parameters
nsteps=1
timestep=0.1
SCFiterations=100
kickstrength=0.1
kickdirection=np.array([1,0,0])
proptype='CFM4'
projectname='CFM4run'

#Grid parameters
L=10.
N_i=60

# Molecule parameters
alpha = np.array([[0.3136497915, 1.158922999, 6.362421394],[0.1688554040, 0.6239137298, 3.425250914]]) #alpha[0] for He, alpha[1] for H
Coef = np.array([0.4446345422, 0.5353281423, 0.1543289673]) #Coefficients are the same for both He and H
R_I = np.array([np.array([0.,0.,0.]), np.array([0.,0.,1.4632])]) #R_I[0] for He, R_I[1] for H.
Z_I = np.array([2.,1.])

t,energies,mu,timings,SE,vNE,EPinit,functioncalls,GridCreatetimes,constructGTOstimes,constructCGFtimes,calculaterealspacedensitytimes,calculatecoredensitytimes,gridintegrationtimes,energycalculationtimes,calculateoverlaptimes,calculatekineticderivativetimes,calculatehartreereciprocaltimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,calculateselfenergytimes,calculateIItimes,DFTsetuptimes,computeE0times,computeDFTtimes,GaussianKicktimes,propagatortimes,propagatetimes,tdttimes,getptimes,LagrangeExtrapolatetimes,popanalysistimes,SEtimes,vNEtimes,PgsEtimes,RTTDDFTtimes,P_initguesstimes,PP_coefstimes=rttddft(nsteps,timestep,proptype,SCFiterations,L,N_i,alpha,Coef,R_I,Z_I,kickstrength,kickdirection,projectname)

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