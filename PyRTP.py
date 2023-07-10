#%%
# Comments correct as 10/7/2023
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
from npy_append_array import NpyAppendArray # Used to save data as external arrays regularly in case of failure to run
import matplotlib.pyplot as plt # Used for post-processing of data
import time # Used to check run times of certain functions
from numba import jit,objmode # Translates Python script into machine code for significantly faster runtimes

# (Advanced users only) The number of threads can be specified by running one of the following functions if required.
# Numpy and Numba will automatically select a number of threads to use, but in the event that less than
# the required number are used, the following can be used to force a number of threads. Run 'np.show_config()'
# to determine which to use.

#os.environ['OMP_NUM_THREADS']='8'
#os.environ['OPENBLAS_NUM_THREADS']='8'
#os.environ['MPI_NUM_THREADS']='8'
#os.environ['MKL_NUM_THREADS']='8'

#%%
# Function Calls
# This section includes all the functions required to perform DFT and RT-TDDFT. The @jit tags instruct Numba to convert
# these functions into machine code, and are cached, so that repeated runs do not have to do this translation on every run.

def GridCreate(L,N_i):
    # This function takes the length of the grid and the number of points per side (N_i) and forms a 3-dimensional 'box' of
    # grid points. This function does not have the @jit tag, as Numba does not support the 'np.linspace' command natively.
    # This command also returns the total number of points and the grid spacing 'dr'.
	N = N_i**3
	dr = (L/N_i)**3

	r_x = np.linspace(-L/2.,L/2.,int(N_i),endpoint=False)
	r_y = np.linspace(-L/2.,L/2.,int(N_i),endpoint=False)
	r_z = np.linspace(-L/2.,L/2.,int(N_i),endpoint=False)
    
	return r_x,r_y,r_z,N,dr


@jit(nopython=True,cache=True)
def GTO(a_GTO,r_GTO,R_GTO): 
    # This function evaluates a Gaussian-type orbital evaluated on the grid from user defined coefficients and exponents, 
    # such that a_GTO = exponent, r_GTO = grid position vector and R_GTO = nuclei position vector. 
    return (2*a_GTO/np.pi)**(3/4)*np.exp(-a_GTO*(np.linalg.norm(r_GTO - R_GTO))**2)

@jit(nopython=True,cache=True)
def construct_GTOs(nuc,N,N_i,r_x,r_y,r_z,R_I,alpha) :
    # This function evaluates the Gaussian-type orbitals for a selected molecule at each point of the grid
    # and saves it to an array. In this case, as the orbital approximation being used is STO-3G, the array
    # has dimensions {3 x N_i x N_i x N_i}, as this approximation uses 3 Gaussian functions per orbital.

    # Create a matrix of grid points for each of the three GTOs, initialised to zero.
    GTO_p = np.zeros(3*N).reshape(3,N_i,N_i,N_i)

    # Currently only working for HeH+, however this would be simple to change for other molecules.
    if nuc == 'He' :
        r_n = R_I[0]
        alpha_n = alpha[0]  
    if nuc == 'H' :
        r_n = R_I[1]
        alpha_n = alpha[1]
        
    #Loop through GTOs and grid points, calculate the GTO value and assign to GTO_p.
    for gto in range(0,3) :
        #for i,j,k in range(0,N_i) :    
        for i in range(0,N_i) : 
            for j in range(0,N_i) :
                for k in range(0,N_i) :
                    p = np.array([r_x[i],r_y[j],r_z[k]]) #Select current grid position vector.

                    GTO_p[gto][i][j][k] = GTO(alpha_n[gto],p,r_n) #calculate GTO value using GTO function call.

    return GTO_p

@jit(nopython=True,cache=True)
def construct_CGF(GTOs,N,N_i,Coef) :
    # This function evaluates the STO-3G approximation at each grid point.
    CGF = np.zeros(N).reshape(N_i,N_i,N_i) #create a matrix of grid points initialised to zero.
    for g in range(0,len(GTOs)) : CGF += Coef[g]*GTOs[g] #construct the CGF from the GTOs and coefficients, Eq. 2.
    
    return CGF

@jit(nopython=True,cache=True)
def calculate_realspace_density(phi,N,N_i,P,dr,) :
    # This function determines the electron charge density at each grid point, along with a total electron charge density.
    n_el_r = np.zeros(N).reshape(N_i,N_i,N_i)
    n_el_total = 0

    for i in range(0,len(phi)) :
        for j in range(0,len(phi)) :
            n_el_r += P[i][j]*phi[i]*phi[j]

    return n_el_r, np.sum(n_el_r)*dr

@jit(nopython=True,cache=True)
def calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I) :
    # This function determines the nuclear charge density at each grid point.
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

    return n_c_r

@jit(nopython=True,cache=True)
def grid_integration(V_r,dr,phi):
    # This function integrates an arbitrary potential over all points of the grid, giving a potential matrix in the
    # correct basis, which is more compact.
    V = np.zeros(len(phi)**2).reshape(len(phi),len(phi))
    
    for i in range(0,len(phi)):
        for j in range(0,len(phi)):

            V[i][j] = np.sum(np.real(V_r)*phi[i]*phi[j])*dr

    return V

@jit(nopython=True,cache=True)
def energy_calculation(V,P): 
    # This function evaluates a potential in the correct basis and determines the corresponding energy contribution. 
    return np.sum(P*V)

@jit(nopython=True,cache=True)
def calculate_overlap(N,N_i,dr,phi):
    # This function evaluates the overlap between atomic orbitals. It is worth noting that this function is simplified using
    # the grid integration function, as if the arbitrary potential being evaluated is equal to 1 at all points, the equations
    # to calculate the potential and the overlap are equivalent. Of course, this could be done explicitly in this function,
    # however it is more compact to do it this way.
    V_temp = np.ones(N).reshape(N_i,N_i,N_i)
    S = grid_integration(V_temp,dr,phi)

    return S

@jit(nopython=True,cache=True)
def calculate_kinetic_derivative(phi,phi_PW_G,N,N_i,G_u,G_v,G_w,L):
    # This function calculates the derivative of the kinetic energy by looping over all points of the reciprocal space grid.
    # 'with objmode()' is used to escape from the machine code to perform functions not available in Numba, with np.matmul being
    # somewhat awkward to use.
    delta_T = np.zeros(len(phi)**2).reshape(len(phi),len(phi))

    for i in range(0,N_i) :
        for j in range(0,N_i) :
            for k in range(0,N_i) :

                g = np.array([G_u[i],G_v[j],G_w[k]])
                for I in range(0,len(phi)) :
                    for J in range(0,len(phi)) :
                        with objmode():
                            delta_T[I][J] += 0.5*L**3/N**2*np.dot(g,g)*np.real(np.dot(np.conjugate(phi_PW_G[I][i][j][k]),phi_PW_G[J][i][j][k]))

    return delta_T

@jit(nopython=True,cache=True)
def calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L):
    # This function calculates the Hartree potential and energy in reciprocal space.
    with objmode(nG='complex128[:,:,:]'):
        nG = np.fft.fftshift(n_G) #n_G is shifted to match same frequency domain as G (-pi,pi) instead of (0,2pi)
    Vg = np.zeros(N).reshape(N_i,N_i,N_i).astype(np.complex128) # Arrays here are of type 'np.complex128', however we 
                                                                # eventually ignore all non-real parts.
    E_hart_G = 0. # Hartree energy in reciprocal space
    
    for i in range(0,N_i) :
        for j in range(0,N_i) :
            for k in range(0,N_i) :

                R_vec = np.array([r_x[i],r_y[j],r_z[k]]) # position vector in real space
                G_vec = np.array([G_u[i],G_v[j],G_w[k]]) # position vector in reciprocal space

                if np.dot(G_vec,G_vec) < 0.01 :  continue # Prevents a division by zero

                Vg[i][j][k] = 4*np.pi*nG[i][j][k]/np.dot(G_vec,G_vec)
                E_hart_G += np.conjugate(nG[i][j][k])*Vg[i][j][k] 
                
    E_hart_G *= L**3/N**2*0.5
    with objmode(Vout='complex128[:,:,:]'):
        Vout=np.fft.ifftshift(Vg)
    return Vout, E_hart_G #result is shifted back. 

@jit(nopython=True,cache=True)
def calculate_hartree_real(V_r,n_r,dr) :
                
    return 0.5*np.sum(V_r*n_r)*dr

def calculate_XC_pylibxc(n_el_r,N_i,dr):
    # This function uses the LibXC library to evaluate the exchange-correlation potential and energy 
    # over the entire grid. This requires the electron charge density to function. The XC functional
    # currently being used is 'LDA_XC_TETER93', which mimics the XC functional used in CP2K.
    func=pylibxc.LibXCFunctional('LDA_XC_TETER93','unpolarized')
    inp={}
    inp["rho"]=n_el_r
    ret=func.compute(n_el_r)
    E_XC_r=ret['zk']
    V_XC_r=ret['vrho']
    V_XC_r = V_XC_r.reshape(N_i,N_i,N_i)
    E_XC_r = E_XC_r.reshape(N_i,N_i,N_i)
    E_XC=np.sum(E_XC_r*n_el_r)*dr

    return V_XC_r, E_XC

@jit(nopython=True,cache=True)
def calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,R_I) :
    # This function calculates the short range pseudopotential, again by looping over all grid points.
    V_SR_r = np.zeros(N).reshape(N_i,N_i,N_i)
    alpha_pp = 5./np.sqrt(2.) #pseudopotential parameter, alpha_pp = 1/R^{c}_{I}
    
    for i in range(0,len(V_SR_r)) :
        for j in range(0,len(V_SR_r)) :
            for k in range(0,len(V_SR_r)) :
                R_vec = np.array([r_x[i],r_y[j],r_z[k]])
                for n in range(0,len(Z_I)) : #loop over nuclei
                    r = np.linalg.norm(R_vec - R_I[n])
                    for c in range(0,len(Cpp)) : #loop over values of Cpp
                        V_SR_r[i][j][k] += Cpp[n][c]*(np.sqrt(2.)*alpha_pp*r)**(2*(c+1)-2)*np.exp(-(alpha_pp*r)**2) 
                        
    return V_SR_r

@jit(nopython=True,cache=True)
def calculate_self_energy(Z_I) :
    # This function calculates the self energy correction term of the total energy, which is relatively simple.
    R_pp = np.sqrt(2.)/5. #R_{I}^{c}

    return np.sum(-(2*np.pi)**(-1/2)*Z_I**2/R_pp)

@jit(nopython=True,cache=True)
def calculate_Ion_interaction(Z_I,R_I) :
    # This function calculates the energy from the interation between ions. 
    R_pp = np.sqrt(2.)/5. #R_{I}^{c}
    E_II = Z_I[0]*Z_I[1]/np.linalg.norm(R_I[0]-R_I[1])*math.erfc(np.linalg.norm(R_I[0]-R_I[1])/np.sqrt(R_pp**2+R_pp**2))

    return E_II

def dftSetup(R_I,alpha,Coef,L,N_i,Z_I):
    # This function calls all functions required to perform DFT that do not change during the SCF cycle.
    # This prevents the functions being called unneccessarily. This function should be called prior to the 
    # 'computeDFT' function.

    r_x,r_y,r_z,N,dr=GridCreate(L,N_i)
    GTOs_He = construct_GTOs('He',N,N_i,r_x,r_y,r_z,R_I,alpha)
    CGF_He = construct_CGF(GTOs_He,N,N_i,Coef)
    GTOs_H = construct_GTOs('H',N,N_i,r_x,r_y,r_z,R_I,alpha)
    CGF_H = construct_CGF(GTOs_H,N,N_i,Coef)
    phi = np.array([CGF_He,CGF_H])
    G_u = np.linspace(-N_i*np.pi/L,N_i*np.pi/L,N_i,endpoint=False)
    G_v = np.linspace(-N_i*np.pi/L,N_i*np.pi/L,N_i,endpoint=False)
    G_w = np.linspace(-N_i*np.pi/L,N_i*np.pi/L,N_i,endpoint=False)
    PW_He_G = np.fft.fftshift(np.fft.fftn(CGF_He))
    PW_H_G = np.fft.fftshift(np.fft.fftn(CGF_H))
    phi_PW_G = np.array([PW_He_G,PW_H_G])
    S = calculate_overlap(N,N_i,dr,phi)
    delta_T = calculate_kinetic_derivative(phi,phi_PW_G,N,N_i,G_u,G_v,G_w,L)
    E_self = calculate_self_energy(Z_I)
    E_II = calculate_Ion_interaction(Z_I,R_I)

    return r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi


def computeE_0(R_I,Z_I,P,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi):
    # This function returns the total energy for a given density matrix P.
    n_el_r, n_el_r_tot  = calculate_realspace_density(phi,N,N_i,P,dr)
    n_c_r = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I)
    n_r = n_el_r + n_c_r
    T = energy_calculation(delta_T,P)
    n_G = np.fft.fftn(n_r)
    V_G, E_hart_G = calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L)
    V_r = np.fft.ifftn(V_G)
    V_hart = grid_integration(V_r,dr,phi)
    E_hart_r = calculate_hartree_real(V_r,n_r,dr)
    V_XC_r,E_XC = calculate_XC_pylibxc(n_el_r,N_i,dr)
    V_XC = grid_integration(V_XC_r,dr,phi)
    V_SR_r = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,R_I)
    V_SR = grid_integration(V_SR_r,dr,phi)
    E_SR = energy_calculation(V_SR,P)
    E_0 = E_hart_r + E_XC + E_SR + T + E_self + E_II

    return E_0

def computeDFT(R_I,alpha,Coef,L,N_i,P_init,Z_I,Cpp,iterations,r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi):
    # This function performs conventional ground-state DFT to produce a converged ground state electron density, wavefunction, energy and Kohn-Sham orbitals.
    # The 'DFTsetup' function must be called prior to calling this function.
    P = P_init
    n_el_r, n_el_r_tot  = calculate_realspace_density(phi,N,N_i,P,dr)
    n_c_r = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I)
    n_r = n_el_r + n_c_r
    T = energy_calculation(delta_T,P)
    n_G = np.fft.fftn(n_r)
    V_G, E_hart_G = calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L)
    V_r = np.fft.ifftn(V_G)
    V_hart = grid_integration(np.real(V_r),dr,phi)
    E_hart_r = calculate_hartree_real(V_r,n_r,dr)
    V_XC_r,E_XC = calculate_XC_pylibxc(n_el_r,N_i,dr)
    V_XC = grid_integration(V_XC_r,dr,phi)
    V_SR_r = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,R_I)
    V_SR = grid_integration(V_SR_r,dr,phi)
    E_SR = energy_calculation(V_SR,P)
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
        n_el_r, n_el_r_tot=calculate_realspace_density(phi,N,N_i,P,dr)
        n_c_r = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I)
        n_r = n_el_r + n_c_r
        n_G = np.fft.fftn(n_r)
        V_G, E_hart_G = calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L)
        V_r = np.fft.ifftn(V_G)
        V_hart = grid_integration(np.real(V_r),dr,phi)
        E_hart_r = calculate_hartree_real(V_r,n_r,dr)
        V_XC_r,E_XC = calculate_XC_pylibxc(n_el_r,N_i,dr)
        V_XC = grid_integration(V_XC_r,dr,phi)
        V_SR_r = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,R_I)
        V_SR = grid_integration(V_SR_r,dr,phi)
        E_SR = energy_calculation(V_SR,P)
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
        P_new=np.array([[0., 0.],[0., 0.]])
        for u in range(0,2) :
            for v in range(0,2) :
                for p in range(0,1) :
                    P_new[u][v] += C[u][p]*C[v][p]
                P_new[u][v] *=2
        if abs(P[0][0]-P_new[0][0]) <= err and abs(P[0][1]-P_new[0][1]) <= err and abs(P[1][0]-P_new[1][0]) <= err and abs(P[1][1]-P_new[1][1]) <= err :
            break
                
        P = P_new 
    return P,np.real(E_0),C,KS

# GaussianKick - provides energy to the system in the form of an electric pulse
# with a Gaussian envelop (run each propagation step!)
def GaussianKick(KS,scale,direction,t,r_x,r_y,r_z,CGF_He,CGF_H,dr):
    t0=1 # These parameters, representing the pulse center and width
    w=0.2 # were selected to be ideal for most systems.
             # In future, it is possible that the user could be given some
             # control over these parameters
    Efield=np.dot(scale*np.exp((-(t-t0)**2)/(2*(w**2))),direction)
    D_x,D_y,D_z,D_tot=transition_dipole_tensor_calculation(r_x,r_y,r_z,CGF_He,CGF_H,dr)
    V_app=-(np.dot(D_x,Efield[0])+np.dot(D_y,Efield[1])+np.dot(D_z,Efield[2]))
    KS_new=KS+V_app 
    return KS_new

# Propagator - this function selects the equation to determine the unitary operator to propagate 
# the system forward in time a singular timestep, using predictor-corrector regime (if applicable)
# This function uses a match-case statement, hence the need for Python 3.10 or greater.
def propagator(select,H1,H2,dt): #propagator functions
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

    return U

def propagate(R_I,Z_I,P,H,C,dt,select,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,Hprev,t,energies,tnew,i,phi):
    # This function performs the propagation to determine the new density matrix at the next timestep, depending on the propagator selected.
    match select:
        case 'CN':
            # The Crank-Nicholson propagator uses a predictor-corrector algorithm to ensure the propagation is accurate.
            st=time.time()
            # This is the predicting section
            if i==0:
                H_p=H
            elif i>0 and i<5:
                H_p=LagrangeExtrapolate(t[0:i],energies[0:i],t[i-1]+(1/2)) # LagrangeExtrapolate fits a curve to the previous set of energies
            else:                                                          # and extrapolates based on that. It is important that the number of
                                                                           # points be limited so as not to cause instability.
                H_p=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],t[i-1]+(1/2))   
            U=np.real(propagator(select,H_p,[],dt)) # calling the propagtor function to determine the unitary operator U.
            C_p=np.dot(U,C)
            # This is the updating section
            P_p=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi)
            # This is the correcting section
            H_c=H+(1/2)*(E_0-H)
            U=np.real(propagator(select,H_c,[],dt)) # After the predictor-corrector regime, the unitary operator U is likely more accurate.
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            et=time.time()
        case 'EM':
            # The Exponential Midpoint propagator uses a predictor-corrector algorithm to ensure the propagation is accurate.
            st=time.time()
            # This is the predicting section
            if i==0:
                H_p=H
            elif i>0 and i<5:
                H_p=LagrangeExtrapolate(t[0:i],energies[0:i],t[i-1]+(1/2)) 
            else:
                H_p=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],t[i-1]+(1/2))    
            U=np.real(propagator(select,H_p,[],dt))
            C_p=np.dot(U,C)
            # # This is the updating section
            P_p=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi)
            # This is the correcting section
            H_c=H+(1/2)*(E_0-H)
            U=np.real(propagator(select,H_c,[],dt))
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            et=time.time()
        case 'ETRS':
            # The ETRS propagator requires an approximation of the energy at the next timestep. As such,
            # the EM propagator is used first to determine the energy at the next timestep, then the ETRS
            # propagator is used to ensure the density matrix is accurate and time-reversal symmetry is enforced.
            st=time.time()
            # This is the predicting section of the EM propagator
            if i==0:
                H_p=H
            elif i>0 and i<5:
                H_p=LagrangeExtrapolate(t[0:i],energies[0:i],t[i-1]+(1/2)) 
            else:
                H_p=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],t[i-1]+(1/2))     
            U=np.real(propagator('EM',H_p,[],dt))
            C_p=np.dot(U,C)
            # This is the updating section of the EM propagator
            P_p=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi)
            # This is the correcting section of the EM propagator
            H_c=H+(1/2)*(E_0-H)
            U=np.real(propagator('EM',H_c,[],dt))
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            # As the EM propagator has determined a good approximate of the density matrix at the next timestep,
            # the 'computeE_0' function can be called to determine the energy at the next timestep.
            Hdt=computeE_0(R_I,Z_I,P_new,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi)
            # The ETRS propagator can now be called
            U=np.real(propagator(select,H,Hdt,dt))
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            et=time.time()
        case 'AETRS':
            # The AETRS propagator uses a similar mechanism to the ETRS propagator, except rather than using the EM
            # propagator to predict the new density matrix (and thus energy), Lagrange Extrapolation is used.
            st=time.time()
            if i<5:
                Hdt=LagrangeExtrapolate(t[0:i],energies[0:i],tnew) 
            else:
                Hdt=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],tnew)   
            U=np.real(propagator(select,H,Hdt,dt))
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            et=time.time()
        case 'CAETRS':
            # The CAETRS propagator works much in the same way as the AETRS propagator, with the inclusion of a
            # predictor-corrector algorithm to ensure accuracy, whilst still being significantly more efficient 
            # than the ETRS propagator.
            st=time.time()    
            # This is the predicting section
            if i<5:
                Hdt=LagrangeExtrapolate(t[0:i],energies[0:i],tnew) 
            else:
                Hdt=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],tnew) 
            # This is the updating section
            U=np.real(propagator(select,H,Hdt,dt))
            C_p=np.dot(U,C)
            P_p=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi)
            # This is the correcting section
            H_c=H+(1/2)*(E_0-H)
            Hdt=2*H_c-Hprev
            U=np.real(propagator(select,H_c,Hdt,dt))
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            et=time.time()
        case 'CFM4':
            # The commutator-free 4th order Magnus propagator uses 2 approximations of energies at future times to
            # provide an accurate and efficient approximation of the unitary operator. This method is preferred.
            st=time.time()
            if i<5:
                Ht1=LagrangeExtrapolate(t[0:i],energies[0:i],(t[i-1])+((1/2)-(np.sqrt(3)/6)))
                Ht2=LagrangeExtrapolate(t[0:i],energies[0:i],(t[i-1])+((1/2)+(np.sqrt(3)/6))) 
            else:
                Ht1=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],(t[i-1])+((1/2)-(np.sqrt(3)/6)))
                Ht2=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],(t[i-1])+((1/2)+(np.sqrt(3)/6))) 
            
            U=np.real(propagator(select,Ht1,Ht2,dt))
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            et=time.time()
        case _:
            raise TypeError("Invalid propagator")
    run=et-st
    return P_new,run

@jit(nopython=True,cache=True)
def transition_dipole_tensor_calculation(r_x,r_y,r_z,CGF_He,CGF_H,dr) :
 # This function determines the transition dipole tensor in the x, y and z directions.
 # This is used for calculation of the Gaussian pulse and the dipole moment.
    D_x = [[0.,0.],[0.,0.]]
    for i in range(0,len(r_x)) :
        for j in range(0,len(r_x)) :
            for k in range(0,len(r_x)) :
             
                #Integrate over grid points
                D_x[0][0] += r_x[i]*CGF_He[i][j][k]**2*dr
                D_x[0][1] += r_x[i]*CGF_He[i][j][k]*CGF_H[i][j][k]*dr
                D_x[1][0] += r_x[i]*CGF_H[i][j][k]*CGF_He[i][j][k]*dr
                D_x[1][1] += r_x[i]*CGF_H[i][j][k]**2*dr                

    D_y = [[0.,0.],[0.,0.]]
    for i in range(0,len(r_y)) :
        for j in range(0,len(r_y)) :
            for k in range(0,len(r_y)) :
             
                #Integrate over grid points
                D_y[0][0] += r_y[i]*CGF_He[i][j][k]**2*dr
                D_y[0][1] += r_y[i]*CGF_He[i][j][k]*CGF_H[i][j][k]*dr
                D_y[1][0] += r_y[i]*CGF_H[i][j][k]*CGF_He[i][j][k]*dr
                D_y[1][1] += r_y[i]*CGF_H[i][j][k]**2*dr   

    D_z = [[0.,0.],[0.,0.]]
    for i in range(0,len(r_z)) :
        for j in range(0,len(r_z)) :
            for k in range(0,len(r_z)) :
             
                #Integrate over grid points
                D_z[0][0] += r_z[i]*CGF_He[i][j][k]**2*dr
                D_z[0][1] += r_z[i]*CGF_He[i][j][k]*CGF_H[i][j][k]*dr
                D_z[1][0] += r_z[i]*CGF_H[i][j][k]*CGF_He[i][j][k]*dr
                D_z[1][1] += r_z[i]*CGF_H[i][j][k]**2*dr   

    D_tot=D_x+D_y+D_z

    return D_x,D_y,D_z,D_tot

def GetP(KS,S):
    # This function determines the density matrix for a given Kohn-Sham matrix.
    S=Matrix(S)
    U,s = S.diagonalize()
    s = s**(-0.5)
    X = np.matmul(np.array(U,dtype='float64'),np.array(s,dtype='float64'))
    X_dag = np.matrix.transpose(np.array(X,dtype='float64'))
    KS_temp = Matrix(np.matmul(X_dag,np.matmul(KS,X)))
    C_temp, e = KS_temp.diagonalize()
    C = np.matmul(X,C_temp)
    P_new=np.array([[0., 0.],[0., 0.]])
    for u in range(0,2):
        for v in range(0,2):
            for p in range(0,1) :
                P_new[u][v] += C[u][p]*C[v][p]
            P_new[u][v] *=2

    return P_new

def LagrangeExtrapolate(t,H,tnew):
    # This function performs Lagrange extrapolation.
    f=lagrange(t,H)
    return np.real(f(tnew))

@jit(nopython=True,cache=True)
def pop_analysis(P,S):
    # This function provides Mulliken population analysis for HeH+, however this function will break for larger molecules.
    PS = np.matmul(P,S)
    pop_total = np.trace(PS)
    pop_He = PS[0,0]
    pop_H = PS[1,1]
    return pop_total, pop_He, pop_H

def ShannonEntropy(P):
    # This function determines the Shannon entropy, which has a variety of quantum chemistry uses.
    P_eigvals=np.linalg.eigvals(P)
    P_eigvals_corrected=[x for x in P_eigvals if x>0.00001] # The eigenvalues are tidied to remove small values, as
    P_eigvecbasis=np.diag(P_eigvals_corrected)              # computing the natural log of small values gives inf.
    SE=np.trace(np.dot(P_eigvecbasis,np.log(P_eigvecbasis)))

    return SE

def vonNeumannEntropy(P):
    # This function determines the von Neumann entropy, which represents the 'distance' from a pure state in this program.
    vNE=np.trace(np.dot(P,np.log(P)))
    return vNE

def EntropyPInitBasis(P,Pini):
    # This function produces an entropy-like measure based on the initial density matrix.
    PPinitBasis=np.matmul(np.linalg.inv(Pini),np.matmul(P,Pini))
    EPinitBasis=np.trace(np.dot(np.abs(PPinitBasis),np.log(np.abs(PPinitBasis))))
    return EPinitBasis


def rttddft(nsteps,dt,propagator,SCFiterations,L,N_i,alpha,Coef,R_I,Z_I,Cpp,P_init,kickstrength,kickdirection,projectname):
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
    'Date of last edit: 10/07/2023\n'+
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
    # Performing calculation of all constant variables to remove the need to repeat it multiple times.
    r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi=dftSetup(R_I,alpha,Coef,L,N_i,Z_I)

    # Compute the ground state first
    P,H,C,KS=computeDFT(R_I,alpha,Coef,L,N_i,P_init,Z_I,Cpp,SCFiterations,r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi)
    # initialising all variable arrays
    Pgs=P
    energies=[]
    mu=[]
    propagationtimes=[]
    SE=[]
    vNE=[]
    EPinit=[]
    # Writing a short .txt file giving the simulation parameters to the project folder
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
        KS=GaussianKick(KS,kickstrength,kickdirection,t[i],r_x,r_y,r_z,CGF_He,CGF_H,dr)
        #Getting perturbed density matrix
        P=GetP(KS,S)
        # Propagating depending on method.
        # AETRS, CAETRS and CFM4 require 2 prior timesteps, which use ETRS
        if i<2 and propagator==('AETRS'):
            P,proptime=propagate(R_I,Z_I,P,H,C,dt,'ETRS',N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i,phi)
        elif i<2 and propagator==('CAETRS'):
            P,proptime=propagate(R_I,Z_I,P,H,C,dt,'ETRS',N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i,phi)
        elif i<2 and propagator==('CFM4'):
            P,proptime=propagate(R_I,Z_I,P,H,C,dt,'ETRS',N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i,phi)
        elif i>1 and propagator==('AETRS'):
            P,proptime=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,energies[i-1],t,energies,t[i],i,phi)
        elif i>1 and propagator==('CAETRS'):
            P,proptime=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,energies[i-1],t,energies,t[i],i,phi)
        elif i>1 and propagator==('CFM4'):
            P,proptime=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,energies[i-1],t,energies,t[i],i,phi)
        else:
            P,proptime=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i,phi)
        print('Propagation time: '+str(proptime))
        # Converging on accurate KS and P
        P,H,C,KS=computeDFT(R_I,alpha,Coef,L,N_i,P,Z_I,Cpp,SCFiterations,r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi)
        # Information Collection
        D_x,D_y,D_z,D_tot=transition_dipole_tensor_calculation(r_x,r_y,r_z,CGF_He,CGF_H,dr)
        mu_t=np.trace(np.dot(D_tot,P))
        energies.append(H)
        SE.append(ShannonEntropy(P))
        vNE.append(vonNeumannEntropy(P))
        EPinit.append(EntropyPInitBasis(P,Pgs))
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
    
    return t,energies,mu,propagationtimes,SE,vNE,EPinit

#%%
# Simulation parameters
nsteps=100
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
Cpp = np.array([[-9.14737128,1.71197792],[-4.19596147,0.73049821]]) #He, H
P_init=np.array([[1.333218,0.],[0.,0.666609]])

t,energies,mu,timings,SE,vNE,EPinit=rttddft(nsteps,timestep,proptype,SCFiterations,L,N_i,alpha,Coef,R_I,Z_I,Cpp,P_init,kickstrength,kickdirection,projectname)

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
