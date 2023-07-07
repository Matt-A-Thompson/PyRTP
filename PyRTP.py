#%%
#Package imports
import numpy as np
from npy_append_array import NpyAppendArray #used to save data as external arrays regularly in case of failure to run
import math
import scipy
import matplotlib.pyplot as plt
from sympy import Matrix
from scipy.interpolate import lagrange
import time
import os
import pylibxc
from numba import jit,objmode

# the number of threads can be specified by uncommenting one of the following functions,
# run 'np.show_config()' to determine which to use.
#os.environ['OMP_NUM_THREADS']='8'
#os.environ['OPENBLAS_NUM_THREADS']='8'
#os.environ['MPI_NUM_THREADS']='8'
#os.environ['MKL_NUM_THREADS']='8'

#%%
# FunctionCalls
def GridCreate(L,N_i,functioncalls,GridCreatetimes):
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

#a_GTO = exponent, r_GTO = grid position vector, R_GTO = nuclei position vector
@jit(nopython=True,cache=True)
def GTO(a_GTO,r_GTO,R_GTO): 
    GTOcalc=(2*a_GTO/np.pi)**(3/4)*np.exp(-a_GTO*(np.linalg.norm(r_GTO - R_GTO))**2)
    return GTOcalc

@jit(nopython=True,cache=True)
def construct_GTOs(nuc,N,N_i,r_x,r_y,r_z,R_I,alpha,functioncalls,constructGTOstimes) :
    with objmode(st='f8'):
        st=time.time()
    #create a matrix of grid points for each of the three GTOs, initialised to zero.
    GTO_p = np.zeros(3*N).reshape(3,N_i,N_i,N_i)

    # Currently only working for HeH+, however this would be simple to change for other molecules
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
    with objmode(et='f8'):
         et=time.time()
    functioncalls[1]+=1
    constructGTOstimes=np.append(constructGTOstimes,et-st)
    return GTO_p,functioncalls,constructGTOstimes

@jit(nopython=True,cache=True)
def construct_CGF(GTOs,N,N_i,Coef,functioncalls,constructCGFtimes) :
    with objmode(st='f8'):
        st=time.time()
    CGF = np.zeros(N).reshape(N_i,N_i,N_i) #create a matrix of grid points initialised to zero.
    for g in range(0,len(GTOs)) : CGF += Coef[g]*GTOs[g] #construct the CGF from the GTOs and coefficients, Eq. 2.

    with objmode(et='f8'):
         et=time.time()
    functioncalls[2]+=1
    constructCGFtimes=np.append(constructCGFtimes,et-st)
    return CGF,functioncalls,constructCGFtimes

@jit(nopython=True,cache=True)
def calculate_realspace_density(phi,N,N_i,P,dr,functioncalls,calculaterealspacedensitytimes) :
    with objmode(st='f8'):
        st=time.time()
    n_el_r = np.zeros(N).reshape(N_i,N_i,N_i)
    n_el_total = 0

    for i in range(0,len(phi)) :
        for j in range(0,len(phi)) :
            n_el_r += P[i][j]*phi[i]*phi[j]

    with objmode(et='f8'):
         et=time.time()
    functioncalls[3]+=1
    calculaterealspacedensitytimes=np.append(calculaterealspacedensitytimes,et-st)
    return n_el_r, np.sum(n_el_r)*dr,functioncalls,calculaterealspacedensitytimes

@jit(nopython=True,cache=True)
def calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I,functioncalls,calculatecoredensitytimes) :
    with objmode(st='f8'):
        st=time.time()
    R_pp = np.sqrt(2.)/5.

    n_c_r = np.zeros(N).reshape(N_i,N_i,N_i)

    for i in range(0,N_i) :
        for j in range(0,N_i) :
            for k in range(0,N_i) :
                r = np.array([r_x[i],r_y[j],r_z[k]])

                for n in range(0,len(Z_I)) :
                    n_c_r[i][j][k] += -Z_I[n]/(R_pp**3)*np.pi**(-3/2)*np.exp(-((r-R_I[n])/R_pp).dot((r-R_I[n])/R_pp))
   
    with objmode(et='f8'):
         et=time.time()
    functioncalls[4]+=1
    calculatecoredensitytimes=np.append(calculatecoredensitytimes,et-st)
    return n_c_r,functioncalls,calculatecoredensitytimes

@jit(nopython=True,cache=True)
def grid_integration(V_r,dr,phi,functioncalls,gridintegrationtimes):
    with objmode(st='f8'):
        st=time.time()
    V = np.zeros(len(phi)**2).reshape(len(phi),len(phi))
    # Numba here
    for i in range(0,len(phi)):
        for j in range(0,len(phi)):

            V[i][j] = np.sum(np.real(V_r)*phi[i]*phi[j])*dr

    with objmode(et='f8'):
         et=time.time()
    functioncalls[5]+=1
    gridintegrationtimes=np.append(gridintegrationtimes,et-st)
    return V,functioncalls,gridintegrationtimes

@jit(nopython=True,cache=True)
def energy_calculation(V,P,functioncalls,energycalculationtimes): 
    with objmode(st='f8'):
        st=time.time()
    E=np.sum(P*V)
    with objmode(et='f8'):
         et=time.time()
    functioncalls[6]+=1
    energycalculationtimes=np.append(energycalculationtimes,et-st)
    return E,functioncalls,energycalculationtimes

@jit(nopython=True,cache=True)
def calculate_overlap(N,N_i,dr,phi,functioncalls,calculateoverlaptimes,gridintegrationtimes):
    with objmode(st='f8'):
        st=time.time()
    V_temp = np.ones(N).reshape(N_i,N_i,N_i)
    S,functioncalls,gridintegrationtimes = grid_integration(V_temp,dr,phi,functioncalls,gridintegrationtimes)
    with objmode(et='f8'):
         et=time.time()
    functioncalls[7]+=1
    calculateoverlaptimes=np.append(calculateoverlaptimes,et-st)
    
    #print(functioncalls)
    #print(calculateoverlaptimes)
    #print(gridintegrationtimes)
    return S,functioncalls,calculateoverlaptimes,gridintegrationtimes

@jit(nopython=True,cache=True)
def calculate_kinetic_derivative(phi,phi_PW_G,N,N_i,G_u,G_v,G_w,L,functioncalls,calculatekineticderivativetimes) :
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
    functioncalls[8]+=1
    calculatekineticderivativetimes=np.append(calculatekineticderivativetimes,et-st)
    return delta_T,functioncalls,calculatekineticderivativetimes

@jit(nopython=True,cache=True)
def calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L,functioncalls,calculatehartreereciprocaltimes):
    with objmode(st='f8',nG='complex128[:,:,:]'):
        st=time.time()
        nG = np.fft.fftshift(n_G) #n_G is shifted to match same frequency domain as G (-pi,pi) instead of (0,2pi)
    Vg = np.zeros(N).reshape(N_i,N_i,N_i).astype(np.complex128)
    E_hart_G = 0. ## Hartree energy in reciprocal space
    
    for i in range(0,N_i) :
        for j in range(0,N_i) :
            for k in range(0,N_i) :

                R_vec = np.array([r_x[i],r_y[j],r_z[k]])
                G_vec = np.array([G_u[i],G_v[j],G_w[k]])

                if np.dot(G_vec,G_vec) < 0.01 :  continue #can't divide by zero

                Vg[i][j][k] = 4*np.pi*nG[i][j][k]/np.dot(G_vec,G_vec)
                E_hart_G += np.conjugate(nG[i][j][k])*Vg[i][j][k] 
                
    E_hart_G *= L**3/N**2*0.5
    with objmode(et='f8',Vout='complex128[:,:,:]'):
         Vout=np.fft.ifftshift(Vg)
         et=time.time()
    functioncalls[9]+=1
    calculatehartreereciprocaltimes=np.append(calculatehartreereciprocaltimes,et-st)
    return Vout, E_hart_G,functioncalls,calculatehartreereciprocaltimes #result is shifted back. 

@jit(nopython=True,cache=True)
def calculate_hartree_real(V_r,n_r,dr,functioncalls,calculatehartreerealtimes) :
    with objmode(st='f8'):
        st=time.time()
    Har=0.5*np.sum(V_r*n_r)*dr
    with objmode(et='f8'):
         et=time.time()
    functioncalls[10]+=1
    calculatehartreerealtimes=np.append(calculatehartreerealtimes,et-st)
    return Har,functioncalls,calculatehartreerealtimes

def calculate_XC_pylibxc(n_el_r,N_i,dr,functioncalls,calculateXCtimes):
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
    functioncalls[11]+=1
    calculateXCtimes=np.append(calculateXCtimes,et-st)
    return V_XC_r, E_XC,functioncalls,calculateXCtimes

@jit(nopython=True,cache=True)
def calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,R_I,functioncalls,calculateVSRtimes) :
    with objmode(st='f8'):
        st=time.time()
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

    with objmode(et='f8'):
         et=time.time()
    functioncalls[12]+=1
    calculateVSRtimes=np.append(calculateVSRtimes,et-st)     
    return V_SR_r,functioncalls,calculateVSRtimes

@jit(nopython=True,cache=True)
def calculate_self_energy(Z_I,functioncalls,calculateselfenergytimes) :
    with objmode(st='f8'):
        st=time.time()
    R_pp = np.sqrt(2.)/5. #R_{I}^{c}
    self=np.sum(-(2*np.pi)**(-1/2)*Z_I**2/R_pp)
    with objmode(et='f8'):
         et=time.time()
    functioncalls[13]+=1
    calculateselfenergytimes=np.append(calculateselfenergytimes,et-st)
    return self,functioncalls,calculateselfenergytimes

@jit(nopython=True,cache=True)
def calculate_Ion_interaction(Z_I,R_I,functioncalls,calculateIItimes) :
    with objmode(st='f8'):
        st=time.time()
    R_pp = np.sqrt(2.)/5. #R_{I}^{c}

    E_II = Z_I[0]*Z_I[1]/np.linalg.norm(R_I[0]-R_I[1])*math.erfc(np.linalg.norm(R_I[0]-R_I[1])/np.sqrt(R_pp**2+R_pp**2))
    with objmode(et='f8'):
         et=time.time()
    functioncalls[14]+=1
    calculateIItimes=np.append(calculateIItimes,et-st)
    return E_II,functioncalls,calculateIItimes

def dftSetup(R_I,alpha,Coef,L,N_i,Z_I,functioncalls,GridCreatetimes,constructGTOstimes,constructCGFtimes,calculateoverlaptimes,gridintegrationtimes,calculatekineticderivativetimes,calculateselfenergytimes,calculateIItimes,DFTsetuptimes):
    with objmode(st='f8'):
        st=time.time()
    r_x,r_y,r_z,N,dr,functioncalls,GridCreatetimes=GridCreate(L,N_i,functioncalls,GridCreatetimes)
    GTOs_He,functioncalls,constructGTOstimes = construct_GTOs('He',N,N_i,r_x,r_y,r_z,R_I,alpha,functioncalls,constructGTOstimes)
    CGF_He,functioncalls,constructCGFtimes = construct_CGF(GTOs_He,N,N_i,Coef,functioncalls,constructCGFtimes)
    GTOs_H,functioncalls,constructGTOstimes = construct_GTOs('H',N,N_i,r_x,r_y,r_z,R_I,alpha,functioncalls,constructGTOstimes)
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
    E_self,functioncalls,calculateselfenergytimes = calculate_self_energy(Z_I,functioncalls,calculateselfenergytimes)
    E_II,functioncalls,calculateIItimes = calculate_Ion_interaction(Z_I,R_I,functioncalls,calculateIItimes)
    with objmode(et='f8'):
         et=time.time()
    functioncalls[15]+=1
    DFTsetuptimes=np.append(DFTsetuptimes,et-st)
    return r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi,functioncalls,GridCreatetimes,constructGTOstimes,constructCGFtimes,calculateoverlaptimes,gridintegrationtimes,calculatekineticderivativetimes,calculateselfenergytimes,calculateIItimes,DFTsetuptimes


def computeE_0(R_I,Z_I,P,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times):
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
    V_SR_r,functioncalls,calculateVSRtimes = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,R_I,functioncalls,calculateVSRtimes)
    V_SR,functioncalls,gridintegrationtimes = grid_integration(V_SR_r,dr,phi,functioncalls,gridintegrationtimes)
    E_SR,functioncalls,energycalculationtimes = energy_calculation(V_SR,P,functioncalls,energycalculationtimes)
    E_0 = E_hart_r + E_XC + E_SR + T + E_self + E_II
    with objmode(et='f8'):
         et=time.time()
    functioncalls[16]+=1
    computeE0times=np.append(computeE0times,et-st)
    return E_0,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times

def computeDFT(R_I,alpha,Coef,L,N_i,P_init,Z_I,Cpp,iterations,r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeDFTtimes):
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
    V_SR_r,functioncalls,calculateVSRtimes = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,R_I,functioncalls,calculateVSRtimes)
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
        V_SR_r,functioncalls,calculateVSRtimes = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,R_I,functioncalls,calculateVSRtimes)
        V_SR,functioncalls,gridintegrationtimes = grid_integration(V_SR_r,dr,phi,functioncalls,gridintegrationtimes)
        E_SR,functioncalls,energycalculationtimes = energy_calculation(V_SR,P,functioncalls,energycalculationtimes)
        E_0 = E_hart_r + E_XC + E_SR + T + E_self + E_II
        KS = np.array(delta_T)+np.array(V_hart)+np.array(V_SR)+np.array(V_XC)
        KS = np.real(KS)
        KS_temp = Matrix(np.matmul(X_dag,np.matmul(KS,X)))
        C_temp, e = KS_temp.diagonalize()
        C = np.matmul(X,C_temp)
        print("iteration : ", I+1, "\n")
        print("total energy : ", np.real(E_0), "\n")
        print("density matrix : ","\n", P, "\n")
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
    with objmode(et='f8'):
         et=time.time()
    functioncalls[17]+=1
    computeDFTtimes=np.append(computeDFTtimes,et-st)
    return P,np.real(E_0),C,KS,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeDFTtimes

# GaussianKick - provides energy to the system in the form of an electric pulse
# with a Gaussian envelop (run each propagation step!)
def GaussianKick(KS,scale,direction,t,r_x,r_y,r_z,CGF_He,CGF_H,dr,functioncalls,GaussianKicktimes,tdttimes):
    with objmode(st='f8'):
        st=time.time()
    t0=1
    w=0.2
    Efield=np.dot(scale*np.exp((-(t-t0)**2)/(2*(w**2))),direction)
    D_x,D_y,D_z,D_tot,functioncalls,tdttimes=transition_dipole_tensor_calculation(r_x,r_y,r_z,CGF_He,CGF_H,dr,functioncalls,tdttimes)
    V_app=-(np.dot(D_x,Efield[0])+np.dot(D_y,Efield[1])+np.dot(D_z,Efield[2]))
    KS_new=KS+V_app
    with objmode(et='f8'):
         et=time.time()
    functioncalls[18]+=1
    GaussianKicktimes=np.append(GaussianKicktimes,et-st)
    return KS_new,functioncalls,GaussianKicktimes,tdttimes

# Propagator - using predictor-corrector regime (if applicable)
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
    functioncalls[19]+=1
    propagatortimes=np.append(propagatortimes,et-st)
    return U,functioncalls,propagatortimes

def propagate(R_I,Z_I,P,H,C,dt,select,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,Hprev,t,energies,tnew,i,phi,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times):
    match select:
        case 'CN':
            with objmode(st='f8'):
                st=time.time()
            # predictor step first
            if i==0:
                H_p=H
            elif i>0 and i<5:
                H_p,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[0:i],energies[0:i],t[i-1]+(1/2),functioncalls,LagrangeExtrapolatetimes) 
            else:
                H_p,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],t[i-1]+(1/2),functioncalls,LagrangeExtrapolatetimes)   
            U,functioncalls,propagatortimes=propagator(select,H_p,[],dt,functioncalls,propagatortimes)
            U=np.real(U)
            C_p=np.dot(U,C)
            # update
            P_p=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
            # correct
            H_c=H+(1/2)*(E_0-H)
            U,functioncalls,propagatortimes=propagator(select,H_c,[],dt,functioncalls,propagatortimes)
            U=np.real(U)
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            with objmode(et='f8'):
                et=time.time()
        case 'EM':
            with objmode(st='f8'):
                st=time.time()
            # predictor step first
            if i==0:
                H_p=H
            elif i>0 and i<5:
                H_p,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[0:i],energies[0:i],t[i-1]+(1/2),functioncalls,LagrangeExtrapolatetimes)  
            else:
                H_p,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],t[i-1]+(1/2),functioncalls,LagrangeExtrapolatetimes)    
            U,functioncalls,propagatortimes=propagator(select,H_p,[],dt,functioncalls,propagatortimes)
            U=np.real(U)
            C_p=np.dot(U,C)
            # update
            P_p=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
            H_c=H+(1/2)*(E_0-H)
            U,functioncalls,propagatortimes=propagator(select,H_c,[],dt,functioncalls,propagatortimes)
            U=np.real(U)
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            with objmode(et='f8'):
                et=time.time()
        case 'ETRS':
            with objmode(st='f8'):
                st=time.time()
            if i==0:
                H_p=H
            elif i>0 and i<5:
                H_p,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[0:i],energies[0:i],t[i-1]+(1/2),functioncalls,LagrangeExtrapolatetimes) 
            else:
                H_p,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],t[i-1]+(1/2),functioncalls,LagrangeExtrapolatetimes)     
            U,functioncalls,propagatortimes=propagator('EM',H_p,[],dt,functioncalls,propagatortimes)
            U=np.real(U)
            C_p=np.dot(U,C)
            # update
            P_p=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
            H_c=H+(1/2)*(E_0-H)
            U,functioncalls,propagatortimes=propagator('EM',H_c,[],dt,functioncalls,propagatortimes)
            U=np.real(U)
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            Hdt,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=computeE_0(R_I,Z_I,P_new,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
            U,functioncalls,propagatortimes=propagator(select,H,Hdt,dt,functioncalls,propagatortimes)
            U=np.real(U)
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            with objmode(et='f8'):
                et=time.time()
        case 'AETRS':
            with objmode(st='f8'):
                st=time.time()
            if i<5:
                Hdt,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[0:i],energies[0:i],tnew,functioncalls,LagrangeExtrapolatetimes) 
            else:
                Hdt,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],tnew,functioncalls,LagrangeExtrapolatetimes)   
            U,functioncalls,propagatortimes=propagator(select,H,Hdt,dt,functioncalls,propagatortimes)
            U=np.real(U)
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            with objmode(et='f8'):
                et=time.time()
        case 'CAETRS':
            with objmode(st='f8'):
                st=time.time()    
            if i<5:
                Hdt,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[0:i],energies[0:i],tnew,functioncalls,LagrangeExtrapolatetimes) 
            else:
                Hdt,functioncalls,LagrangeExtrapolatetimes=LagrangeExtrapolate(t[i-5:i],energies[i-5:i],tnew,functioncalls,LagrangeExtrapolatetimes) 
            U,functioncalls,propagatortimes=propagator(select,H,Hdt,dt,functioncalls,propagatortimes)
            U=np.real(U)
            C_p=np.dot(U,C)
            P_p=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,phi,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
            H_c=H+(1/2)*(E_0-H)
            Hdt=2*H_c-Hprev
            U,functioncalls,propagatortimes=propagator(select,H_c,Hdt,dt,functioncalls,propagatortimes)
            U=np.real(U)
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            with objmode(et='f8'):
                et=time.time()
        case 'CFM4':
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
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            with objmode(et='f8'):
                et=time.time()
        case _:
            raise TypeError("Invalid propagator")
    functioncalls[20]+=1
    propagatetimes=np.append(propagatetimes,et-st)
    return P_new,et-st,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times

@jit(nopython=True,cache=True)
def transition_dipole_tensor_calculation(r_x,r_y,r_z,CGF_He,CGF_H,dr,functioncalls,tdttimes) :
    with objmode(st='f8'):
        st=time.time()
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
    with objmode(et='f8'):
         et=time.time()
    functioncalls[21]+=1
    tdttimes=np.append(tdttimes,et-st)
    return D_x,D_y,D_z,D_tot,functioncalls,tdttimes

def GetP(KS,S,functioncalls,getptimes):
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
    P_new=np.array([[0., 0.],[0., 0.]])
    for u in range(0,2):
        for v in range(0,2):
            for p in range(0,1) :
                P_new[u][v] += C[u][p]*C[v][p]
            P_new[u][v] *=2
    with objmode(et='f8'):
         et=time.time()
    functioncalls[22]+=1
    getptimes=np.append(getptimes,et-st)
    return P_new,functioncalls,getptimes

def LagrangeExtrapolate(t,H,tnew,functioncalls,LagrangeExtrapolatetimes):
    with objmode(st='f8'):
        st=time.time()
    f=lagrange(t,H)
    val=np.real(f(tnew))
    with objmode(et='f8'):
         et=time.time()
    functioncalls[23]+=1
    LagrangeExtrapolatetimes=np.append(LagrangeExtrapolatetimes,et-st)
    return val,functioncalls,LagrangeExtrapolatetimes

@jit(nopython=True,cache=True)
def pop_analysis(P,S,functioncalls,popanalysistimes) :
    with objmode(st='f8'):
        st=time.time()
    PS = np.matmul(P,S)
    pop_total = np.trace(PS)
    pop_He = PS[0,0]
    pop_H = PS[1,1]
    with objmode(et='f8'):
         et=time.time()
    functioncalls[24]+=1
    popanalysistimes=np.append(popanalysistimes,et-st)
    return pop_total, pop_He, pop_H,functioncalls,popanalysistimes

def ShannonEntropy(P,functioncalls,SEtimes):
    with objmode(st='f8'):
        st=time.time()
    P_eigvals=np.linalg.eigvals(P)
    P_eigvals_corrected=[x for x in P_eigvals if x>0.00001]
    P_eigvecbasis=np.diag(P_eigvals_corrected)
    SE=np.trace(np.dot(P_eigvecbasis,np.log(P_eigvecbasis)))
    with objmode(et='f8'):
         et=time.time()
    functioncalls[25]+=1
    SEtimes=np.append(SEtimes,et-st)
    return SE,functioncalls,SEtimes

def vonNeumannEntropy(P,functioncalls,vNEtimes):
    with objmode(st='f8'):
        st=time.time()
    vNE=np.trace(np.dot(P,np.log(P)))
    with objmode(et='f8'):
         et=time.time()
    functioncalls[26]+=1
    vNEtimes=np.append(vNEtimes,et-st)
    return vNE,functioncalls,vNEtimes

def EntropyPInitBasis(P,Pini,functioncalls,PgsEtimes):
    with objmode(st='f8'):
        st=time.time()
    PPinitBasis=np.matmul(np.linalg.inv(Pini),np.matmul(P,Pini))
    EPinitBasis=np.trace(np.dot(np.abs(PPinitBasis),np.log(np.abs(PPinitBasis))))
    with objmode(et='f8'):
         et=time.time()
    functioncalls[27]+=1
    PgsEtimes=np.append(PgsEtimes,et-st)
    return EPinitBasis,functioncalls,PgsEtimes


def rttddft(nsteps,dt,propagator,SCFiterations,L,N_i,alpha,Coef,R_I,Z_I,Cpp,P_init,kickstrength,kickdirection,projectname):
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
    'Date of last edit: 06/07/2023\n'+
    'Description: A program to perform RT-TDDFT exclusively in Python.\n'+ 
    '\t     A variety of propagators (CN, EM, ETRS, etc.) can be \n'+
    '\t     selected for benchmarking and testing. Comments on \n'+
    '\t     the approaches used are in progress.\n'+
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
    GridCreatetimes=np.array([])
    constructGTOstimes=np.array([])
    constructCGFtimes=np.array([])
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
    RTTDDFTtimes=np.array([])

    functioncalls=np.zeros((29,),dtype=int)
    # Performing calculation of all constant variables to remove the need to repeat it multiple times.
    r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi,functioncalls,GridCreatetimes,constructGTOstimes,constructCGFtimes,calculateoverlaptimes,gridintegrationtimes,calculatekineticderivativetimes,calculateselfenergytimes,calculateIItimes,DFTsetuptimes=dftSetup(R_I,alpha,Coef,L,N_i,Z_I,functioncalls,GridCreatetimes,constructGTOstimes,constructCGFtimes,calculateoverlaptimes,gridintegrationtimes,calculatekineticderivativetimes,calculateselfenergytimes,calculateIItimes,DFTsetuptimes)

    # Compute the ground state first
    P,H,C,KS,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeDFTtimes=computeDFT(R_I,alpha,Coef,L,N_i,P_init,Z_I,Cpp,SCFiterations,r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeDFTtimes)
    # initialising all variable arrays
    Pgs=P
    energies=[]
    mu=[]
    propagationtimes=[]
    SE=[]
    vNE=[]
    EPinit=[]
    # Writing a short .txt file giving the simulation parameters to the project folder
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
        KS,functioncalls,GaussianKicktimes,tdttimes=GaussianKick(KS,kickstrength,kickdirection,t[i],r_x,r_y,r_z,CGF_He,CGF_H,dr,functioncalls,GaussianKicktimes,tdttimes)
        #Getting perturbed density matrix
        P,functioncalls,getptimes=GetP(KS,S,functioncalls,getptimes)
        # Propagating depending on method.
        # AETRS, CAETRS and CFM4 require 2 prior timesteps, which use ETRS
        if i<2 and propagator==('AETRS'):
            P,proptime,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=propagate(R_I,Z_I,P,H,C,dt,'ETRS',N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i,phi,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
        elif i<2 and propagator==('CAETRS'):
            P,proptime,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=propagate(R_I,Z_I,P,H,C,dt,'ETRS',N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i,phi,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
        elif i<2 and propagator==('CFM4'):
            P,proptime,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=propagate(R_I,Z_I,P,H,C,dt,'ETRS',N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i,phi,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
        elif i>1 and propagator==('AETRS'):
            P,proptime,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,energies[i-1],t,energies,t[i],i,phi,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
        elif i>1 and propagator==('CAETRS'):
            P,proptime,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,energies[i-1],t,energies,t[i],i,phi,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
        elif i>1 and propagator==('CFM4'):
            P,proptime,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,energies[i-1],t,energies,t[i],i,phi,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
        else:
            P,proptime,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i,phi,functioncalls,propagatetimes,propagatortimes,LagrangeExtrapolatetimes,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeE0times)
        print('Propagation time: '+str(proptime))
        # Converging on accurate KS and P
        P,H,C,KS,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeDFTtimes=computeDFT(R_I,alpha,Coef,L,N_i,P,Z_I,Cpp,SCFiterations,r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II,phi,functioncalls,calculaterealspacedensitytimes,calculatecoredensitytimes,energycalculationtimes,calculatehartreereciprocaltimes,gridintegrationtimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,computeDFTtimes)
        # Information Collection
        D_x,D_y,D_z,D_tot,functioncalls,tdttimes=transition_dipole_tensor_calculation(r_x,r_y,r_z,CGF_He,CGF_H,dr,functioncalls,tdttimes)
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
    functioncalls[28]+=1
    RTTDDFTtimes=np.append(RTTDDFTtimes,et-st)
    
    return t,energies,mu,propagationtimes,SE,vNE,EPinit,functioncalls,GridCreatetimes,constructGTOstimes,constructCGFtimes,calculaterealspacedensitytimes,calculatecoredensitytimes,gridintegrationtimes,energycalculationtimes,calculateoverlaptimes,calculatekineticderivativetimes,calculatehartreereciprocaltimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,calculateselfenergytimes,calculateIItimes,DFTsetuptimes,computeE0times,computeDFTtimes,GaussianKicktimes,propagatortimes,propagatetimes,tdttimes,getptimes,LagrangeExtrapolatetimes,popanalysistimes,SEtimes,vNEtimes,PgsEtimes,RTTDDFTtimes

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

t,energies,mu,timings,SE,vNE,EPinit,functioncalls,GridCreatetimes,constructGTOstimes,constructCGFtimes,calculaterealspacedensitytimes,calculatecoredensitytimes,gridintegrationtimes,energycalculationtimes,calculateoverlaptimes,calculatekineticderivativetimes,calculatehartreereciprocaltimes,calculatehartreerealtimes,calculateXCtimes,calculateVSRtimes,calculateselfenergytimes,calculateIItimes,DFTsetuptimes,computeE0times,computeDFTtimes,GaussianKicktimes,propagatortimes,propagatetimes,tdttimes,getptimes,LagrangeExtrapolatetimes,popanalysistimes,SEtimes,vNEtimes,PgsEtimes,RTTDDFTtimes=rttddft(nsteps,timestep,proptype,SCFiterations,L,N_i,alpha,Coef,R_I,Z_I,Cpp,P_init,kickstrength,kickdirection,projectname)

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
filterpercentage=85
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
print('| '+'calculate_realspace_density'.ljust(30)+'|'+str(functioncalls[3]).center(19)+'|'+ str(np.round(np.mean(calculaterealspacedensitytimes),13)).center(19)+'|')
print('| '+'calculate_core_density'.ljust(30)+'|'+str(functioncalls[4]).center(19)+'|'+ str(np.round(np.mean(calculatecoredensitytimes),13)).center(19)+'|')
print('| '+'grid_integration'.ljust(30)+'|'+str(functioncalls[5]).center(19)+'|'+ str(np.round(np.mean(gridintegrationtimes),13)).center(19)+'|')
print('| '+'energy_calculation'.ljust(30)+'|'+str(functioncalls[6]).center(19)+'|'+ str(np.round(np.mean(energycalculationtimes),13)).center(19)+'|')
print('| '+'calculate_overlap'.ljust(30)+'|'+str(functioncalls[7]).center(19)+'|'+ str(np.round(np.mean(calculateoverlaptimes),13)).center(19)+'|')
print('| '+'calculate_kinetic_derivative'.ljust(30)+'|'+str(functioncalls[8]).center(19)+'|'+ str(np.round(np.mean(calculatekineticderivativetimes),13)).center(19)+'|')
print('| '+'calculate_hartree_reciprocal'.ljust(30)+'|'+str(functioncalls[9]).center(19)+'|'+ str(np.round(np.mean(calculatehartreereciprocaltimes),13)).center(19)+'|')
print('| '+'calculate_hartree_real'.ljust(30)+'|'+str(functioncalls[10]).center(19)+'|'+ str(np.round(np.mean(calculatehartreerealtimes),13)).center(19)+'|')
print('| '+'calculate_XC'.ljust(30)+'|'+str(functioncalls[11]).center(19)+'|'+ str(np.round(np.mean(calculateXCtimes),13)).center(19)+'|')
print('| '+'calculate_V_SR_r'.ljust(30)+'|'+str(functioncalls[12]).center(19)+'|'+ str(np.round(np.mean(calculateVSRtimes),13)).center(19)+'|')
print('| '+'calculate_self_energy'.ljust(30)+'|'+str(functioncalls[13]).center(19)+'|'+ str(np.round(np.mean(calculateselfenergytimes),13)).center(19)+'|')
print('| '+'calculate_Ion_interation'.ljust(30)+'|'+str(functioncalls[14]).center(19)+'|'+ str(np.round(np.mean(calculateIItimes),13)).center(19)+'|')
print('| '+'dftSetup'.ljust(30)+'|'+str(functioncalls[15]).center(19)+'|'+ str(np.round(np.mean(DFTsetuptimes),13)).center(19)+'|')
print('| '+'computeE_0'.ljust(30)+'|'+str(functioncalls[16]).center(19)+'|'+ str(np.round(np.mean(computeE0times),13)).center(19)+'|')
print('| '+'computeDFT'.ljust(30)+'|'+str(functioncalls[17]).center(19)+'|'+ str(np.round(np.mean(computeDFTtimes),13)).center(19)+'|')
print('| '+'GaussianKick'.ljust(30)+'|'+str(functioncalls[18]).center(19)+'|'+ str(np.round(np.mean(GaussianKicktimes),13)).center(19)+'|')
print('| '+'propagator'.ljust(30)+'|'+str(functioncalls[19]).center(19)+'|'+ str(np.round(np.mean(propagatortimes),13)).center(19)+'|')
print('| '+'propagate'.ljust(30)+'|'+str(functioncalls[20]).center(19)+'|'+ str(np.round(np.mean(propagatetimes),13)).center(19)+'|')
print('| '+'transition_dipole_tensor'.ljust(30)+'|'+str(functioncalls[21]).center(19)+'|'+ str(np.round(np.mean(tdttimes),13)).center(19)+'|')
print('| '+'GetP'.ljust(30)+'|'+str(functioncalls[22]).center(19)+'|'+ str(np.round(np.mean(getptimes),13)).center(19)+'|')
print('| '+'LagrangeExtrapolate'.ljust(30)+'|'+str(functioncalls[23]).center(19)+'|'+ str(np.round(np.mean(LagrangeExtrapolatetimes),13)).center(19)+'|')
print('| '+'pop_analysis'.ljust(30)+'|'+str(functioncalls[24]).center(19)+'|'+ str(np.round(np.mean(popanalysistimes),13)).center(19)+'|')
print('| '+'ShannonEntropy'.ljust(30)+'|'+str(functioncalls[25]).center(19)+'|'+ str(np.round(np.mean(SEtimes),13)).center(19)+'|')
print('| '+'vonNeumannEntropy'.ljust(30)+'|'+str(functioncalls[26]).center(19)+'|'+ str(np.round(np.mean(vNEtimes),13)).center(19)+'|')
print('| '+'EntropyPInitBasis'.ljust(30)+'|'+str(functioncalls[27]).center(19)+'|'+ str(np.round(np.mean(PgsEtimes),13)).center(19)+'|')
print('| '+'rttddft'.ljust(30)+'|'+str(functioncalls[28]).center(19)+'|'+ str(np.round(np.mean(RTTDDFTtimes),13)).center(19)+'|')
print(''.ljust(73,'-'))
     # '| Function            | Function calls    | Average run time        |\n'+
     # '---------------------------------------------------------------------\n'+
     # '| GridCreate          | '+str(functioncalls[0]).ljust(20,' ')+'    | '+str(np.mean(GridCreatetimes))+'    |')

# %%