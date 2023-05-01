#%%
#Package imports
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
from pyscf import gto, dft, lib
from sympy import Matrix
from scipy.interpolate import lagrange

#%%
# FunctionCalls - all the functions used in the DFT calculation will be stored here
def GridCreate(L,N_i):
	
	N = N_i**3
	dr = (L/N_i)**3

	r_x = np.linspace(-L/2.,L/2.,int(N_i),endpoint=False)
	r_y = np.linspace(-L/2.,L/2.,int(N_i),endpoint=False)
	r_z = np.linspace(-L/2.,L/2.,int(N_i),endpoint=False)
    
	return r_x,r_y,r_z,N,dr

#a_GTO = exponent, r_GTO = grid position vector, R_GTO = nuclei position vector
def GTO(a_GTO,r_GTO,R_GTO): return (2*a_GTO/np.pi)**(3/4)*np.exp(-a_GTO*(np.linalg.norm(r_GTO - R_GTO))**2)

def construct_GTOs(nuc,N,N_i,r_x,r_y,r_z,R_I,alpha) :

    #create a matrix of grid points for each of the three GTOs, initialised to zero.
    GTO_p = np.zeros(3*N).reshape(3,N_i,N_i,N_i)


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

def construct_CGF(GTOs,N,N_i,Coef) :

    CGF = np.zeros(N).reshape(N_i,N_i,N_i) #create a matrix of grid points initialised to zero.
    for g in range(0,len(GTOs)) : CGF += Coef[g]*GTOs[g] #construct the CGF from the GTOs and coefficients, Eq. 2.
    
    return CGF

def calculate_realspace_density(CGF_0,CGF_1,N,N_i,P,dr) :

    n_el_r = np.zeros(N).reshape(N_i,N_i,N_i)
    n_el_total = 0

    for i in range(0,N_i) :
        for j in range(0,N_i) :
            for k in range(0,N_i) :

                n_el_r[i][j][k] = P[0][0]*CGF_0[i][j][k]**2 + P[0][1]*CGF_0[i][j][k]*CGF_1[i][j][k] +\
                    P[1][0]*CGF_1[i][j][k]*CGF_0[i][j][k] + P[1][1]*CGF_1[i][j][k]**2

                n_el_total += n_el_r[i][j][k]*dr

    return n_el_r, n_el_total

def calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I) :

    R_pp = np.sqrt(2.)/5.

    n_c_r = np.zeros(N).reshape(N_i,N_i,N_i)

    for i in range(0,N_i) :
        for j in range(0,N_i) :
            for k in range(0,N_i) :
                r = np.array([r_x[i],r_y[j],r_z[k]])

                for n in range(0,len(Z_I)) :
                    n_c_r[i][j][k] += -Z_I[n]/(R_pp**3)*np.pi**(-3/2)*np.exp(-((r-R_I[n])/R_pp).dot((r-R_I[n])/R_pp))

    return n_c_r

def plot_ongrid(n,plot_type,x,ymin,ymax,zmin,zmax,L,N_i) :
    step = L/N_i
    ext = [-L/2.+zmin*step-step/2.,-L/2.+zmax*step-step/2.,-L/2.+ymin*step-step/2.,-L/2.+ymax*step-step/2.]
    if plot_type == 'image' :
        plt.imshow(n[x,ymin:ymax,zmin:zmax],origin='lower',extent=ext) 
        plt.colorbar()
    if plot_type == 'contour' :
        plt.contour(n[x,ymin:ymax,zmin:zmax],levels=10,origin='lower',extent=ext)
        plt.colorbar()
    plt.xlabel("r (a.u.)")
    plt.ylabel("r (a.u.)")
    
def grid_integration(V_r,dr,CGF_He,CGF_H) :

    V = [[0.,0.],[0.,0.]]
    
    for i in range(0,len(V_r)) :
        for j in range(0,len(V_r)) :
            for k in range(0,len(V_r)) :
             
                #Integrate over grid points
                V[0][0] += V_r[i][j][k]*CGF_He[i][j][k]**2*dr
                V[0][1] += V_r[i][j][k]*CGF_He[i][j][k]*CGF_H[i][j][k]*dr
                V[1][0] += V_r[i][j][k]*CGF_H[i][j][k]*CGF_He[i][j][k]*dr
                V[1][1] += V_r[i][j][k]*CGF_H[i][j][k]**2*dr                                                                             
    
    return V

def energy_calculation(V,P) :
    
    E = P[0][0]*V[0][0] + P[0][1]*V[0][1] + P[1][0]*V[1][0] + P[1][1]*V[1][1]
    
    return E

def calculate_overlap(N_i,dr,CGF_He,CGF_H) :
    
    S = [[0.,0.],[0.,0.]] #initialise a 2 by 2 matrix of zeros.
    
    #Loop through grid points 
    for i in range(0,N_i) : 
        for j in range(0,N_i) :
            for k in range(0,N_i) :
                #Sum the overlap contributions from each CGF as in Eq. 3 
                S[0][0] += CGF_He[i][j][k]**2*dr
                S[0][1] += CGF_He[i][j][k]*CGF_H[i][j][k]*dr
                S[1][0] += CGF_H[i][j][k]*CGF_He[i][j][k]*dr
                S[1][1] += CGF_H[i][j][k]**2*dr

    return S

def calculate_kinetic_derivative(N,N_i,G_u,G_v,G_w,PW_He_G,PW_H_G,L) :

    delta_T = [[0.,0.],[0.,0.]]

    for i in range(0,N_i) :
        for j in range(0,N_i) :
            for k in range(0,N_i) :

                g = np.array([G_u[i],G_v[j],G_w[k]]) #Select current G vector

                #Calculate Kinetic energy matrix as per Eq. 7
                #np.real is required here as PWs are complex. 
                #The complex component is ~zero after taking PW_He_G^2 but the result still contains a complex term
                
                delta_T[0][0] += 0.5*L**3/N**2*np.dot(g,g)*np.real(np.dot(np.conjugate(PW_He_G[i][j][k]),PW_He_G[i][j][k]))
                delta_T[0][1] += 0.5*L**3/N**2*np.dot(g,g)*np.real(np.dot(np.conjugate(PW_He_G[i][j][k]),PW_H_G[i][j][k]))
                delta_T[1][0] += 0.5*L**3/N**2*np.dot(g,g)*np.real(np.dot(np.conjugate(PW_H_G[i][j][k]),PW_He_G[i][j][k]))
                delta_T[1][1] += 0.5*L**3/N**2*np.dot(g,g)*np.real(np.dot(np.conjugate(PW_H_G[i][j][k]),PW_H_G[i][j][k]))


    return delta_T

def calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L) :

    nG = np.fft.fftshift(n_G) #n_G is shifted to match same frequency domain as G (-pi,pi) instead of (0,2pi)
    Vg = np.complex128(np.zeros(N).reshape(N_i,N_i,N_i))
    E_hart_G = 0. ## Hartree energy in reciprocal space, Eq. ??
    
    for i in range(0,N_i) :
        for j in range(0,N_i) :
            for k in range(0,N_i) :

                R_vec = np.array([r_x[i],r_y[j],r_z[k]])
                G_vec = np.array([G_u[i],G_v[j],G_w[k]])

                if np.dot(G_vec,G_vec) < 0.01 :  continue #can't divide by zero

                Vg[i][j][k] = 4*np.pi*nG[i][j][k]/np.dot(G_vec,G_vec) #Eq. ??
                E_hart_G += np.conjugate(nG[i][j][k])*Vg[i][j][k] 
                
    E_hart_G *= L**3/N**2*0.5
    
    return np.fft.ifftshift(Vg), E_hart_G #result is shifted back. 

def calculate_hartree_real(N_i,V_r,n_r,dr) :
    
    E_hart_r = 0.
    
    for i in range(0,N_i) :
        for j in range(0,N_i) :
            for k in range(0,N_i) :
                E_hart_r += 0.5*V_r[i][j][k]*n_r[i][j][k]*dr
                
    return E_hart_r

def calculate_XC_potential(N,N_i,n_el_r):
    
    V_XC_r = np.zeros(N).reshape(N_i,N_i,N_i)
    
    for i in range(0,N_i) :
        for j in range(0,N_i) :
            #Loop through first two axis and obtain a list of n(r) for the third axis
            V_XC_r[i][j] = dft.xcfun.eval_xc('LDAERF',n_el_r[i][j])[1][0] #get the XC term for each n(r) in the list
            
    return V_XC_r

def calculate_XC_energy(N_i,n_el_r,dr):

    E_XC = 0.

    for i in range(0,N_i) :
        for j in range(0,N_i) :
            XC_ene = dft.xcfun.eval_xc('LDAERF',n_el_r[i][j])[0]
            for k in range(0,N_i) :
                E_XC += XC_ene[k]*n_el_r[i][j][k]*dr
            
    return E_XC

def calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,R_I) :

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

def calculate_self_energy(Z_I) :

    R_pp = np.sqrt(2.)/5. #R_{I}^{c}

    E_self = 0.

    for n in range(0,len(Z_I)) :
        E_self += -(2*np.pi)**(-1/2)*Z_I[n]**2/R_pp

    return E_self

def calculate_Ion_interaction(Z_I,R_I) :

    R_pp = np.sqrt(2.)/5. #R_{I}^{c}

    E_II = Z_I[0]*Z_I[1]/np.linalg.norm(R_I[0]-R_I[1])*math.erfc(np.linalg.norm(R_I[0]-R_I[1])/np.sqrt(R_pp**2+R_pp**2))

    return E_II

def dftSetup(R_I,alpha,Coef,L,N_i,Z_I):
    r_x,r_y,r_z,N,dr=GridCreate(L,N_i)
    GTOs_He = construct_GTOs('He',N,N_i,r_x,r_y,r_z,R_I,alpha)
    CGF_He = construct_CGF(GTOs_He,N,N_i,Coef)
    GTOs_H = construct_GTOs('H',N,N_i,r_x,r_y,r_z,R_I,alpha)
    CGF_H = construct_CGF(GTOs_H,N,N_i,Coef)
    G_u = np.linspace(-N_i*np.pi/L,N_i*np.pi/L,N_i,endpoint=False)
    G_v = np.linspace(-N_i*np.pi/L,N_i*np.pi/L,N_i,endpoint=False)
    G_w = np.linspace(-N_i*np.pi/L,N_i*np.pi/L,N_i,endpoint=False)
    PW_He_G = np.fft.fftshift(np.fft.fftn(CGF_He))
    PW_H_G = np.fft.fftshift(np.fft.fftn(CGF_H))
    S = calculate_overlap(N_i,dr,CGF_He,CGF_H)
    delta_T = calculate_kinetic_derivative(N,N_i,G_u,G_v,G_w,PW_He_G,PW_H_G,L)
    E_self = calculate_self_energy(Z_I)
    E_II = calculate_Ion_interaction(Z_I,R_I)

    return r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II

def computeE_0(R_I,Z_I,P,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L):
    #r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II=dftSetup(R_I,alpha,Coef,L,N_i,Z_I)
    n_el_r, n_el_r_tot  = calculate_realspace_density(CGF_He,CGF_H,N,N_i,P,dr)
    n_c_r = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I)
    n_r = n_el_r + n_c_r
    T = energy_calculation(delta_T,P)
    n_G = np.fft.fftn(n_r)
    V_G, E_hart_G = calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L)
    V_r = np.fft.ifftn(V_G)
    V_hart = grid_integration(V_r,dr,CGF_He,CGF_H)
    E_hart_r = calculate_hartree_real(N_i,V_r,n_r,dr)
    V_XC_r = calculate_XC_potential(N,N_i,n_el_r)
    V_XC = grid_integration(V_XC_r,dr,CGF_He,CGF_H)
    E_XC = calculate_XC_energy(N_i,n_el_r,dr)
    V_SR_r = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,R_I)
    V_SR = grid_integration(V_SR_r,dr,CGF_He,CGF_H)
    E_SR = energy_calculation(V_SR,P)
    E_0 = E_hart_r + E_XC + E_SR + T + E_self + E_II

    return E_0

def computeDFT(R_I,alpha,Coef,L,N_i,P_init,Z_I,Cpp,iterations,r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II):

	P = P_init
	n_el_r, n_el_r_tot  = calculate_realspace_density(CGF_He,CGF_H,N,N_i,P,dr)
	n_c_r = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I)
	n_r = n_el_r + n_c_r
	T = energy_calculation(delta_T,P)
	n_G = np.fft.fftn(n_r)
	V_G, E_hart_G = calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L)
	V_r = np.fft.ifftn(V_G)
	V_hart = grid_integration(V_r,dr,CGF_He,CGF_H)
	E_hart_r = calculate_hartree_real(N_i,V_r,n_r,dr)
	V_XC_r = calculate_XC_potential(N,N_i,n_el_r)
	V_XC = grid_integration(V_XC_r,dr,CGF_He,CGF_H)
	E_XC = calculate_XC_energy(N_i,n_el_r,dr)
	V_SR_r = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,R_I)
	V_SR = grid_integration(V_SR_r,dr,CGF_He,CGF_H)
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
        #density
		n_el_r, n_el_r_tot=calculate_realspace_density(CGF_He,CGF_H,N,N_i,P,dr)
		n_c_r = calculate_core_density(N,N_i,Z_I,r_x,r_y,r_z,R_I)
		n_r = n_el_r + n_c_r
		n_G = np.fft.fftn(n_r)
		V_G, E_hart_G = calculate_hartree_reciprocal(n_G,N,N_i,r_x,r_y,r_z,G_u,G_v,G_w,L)
		V_r = np.fft.ifftn(V_G)
		V_hart = grid_integration(V_r,dr,CGF_He,CGF_H)
		E_hart_r = calculate_hartree_real(N_i,V_r,n_r,dr)
		V_XC_r = calculate_XC_potential(N,N_i,n_el_r)
		V_XC = grid_integration(V_XC_r,dr,CGF_He,CGF_H)
		E_XC = calculate_XC_energy(N_i,n_el_r,dr)
		V_SR_r = calculate_V_SR_r(N,N_i,r_x,r_y,r_z,Z_I,Cpp,R_I)
		V_SR = grid_integration(V_SR_r,dr,CGF_He,CGF_H)
		E_SR = energy_calculation(V_SR,P)
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
	return P,E_0,C,KS

#DeltaKick - run each propagation step!
def deltaKick(KS,scale,direction,t,r_x,r_y,r_z,CGF_He,CGF_H,dr):
    t0=1
    w=0.2
    Efield=np.dot(scale*np.exp((-(t-t0)**2)/(2*(w**2))),direction)
    D_x,D_y,D_z,D_tot=transition_dipole_tensor_calculation(r_x,r_y,r_z,CGF_He,CGF_H,dr)
    V_app=-(np.dot(D_x,Efield[0])+np.dot(D_y,Efield[1])+np.dot(D_z,Efield[2]))
    KS_new=KS+V_app
    return KS_new

# Propagator - using predictor-corrector regime
def propagator(select,H1,H2,dt):
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
            U=np.cross(np.exp(-1j*dt*a1*H1-1j*dt*a2*H2),np.exp(-1j*dt*a2*H1-1j*dt*a1*H2))
        case _:
            raise TypeError("Invalid propagator")

    return U

def propagate(R_I,Z_I,P,H,C,dt,select,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,Hprev,t,energies,tnew,i):
    match select:
        case 'CN':
            # predictor step first
            if i==0:
                H_p=H
            else:
                H_p=LagrangeExtrapolate(t,H,t[i-1]+(1/2))    
            U=np.real(propagator(select,H_p,[],dt))
            C_p=np.dot(U,C)
            # update
            P_p=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L)
            # correct
            H_c=H+(1/2)*(E_0-H)
            U=np.real(propagator(select,H_c,[],dt))
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
        case 'EM':
            # predictor step first
            if i==0:
                H_p=H
            else:
                H_p=LagrangeExtrapolate(t,H,t[i-1]+(1/2))  
            U=np.real(propagator(select,H_p,[],dt))
            C_p=np.dot(U,C)
            # update
            P_p=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L)
            H_c=H+(1/2)*(E_0-H)
            U=np.real(propagator(select,H_c,[],dt))
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
        case 'ETRS':
            if i==0:
                H_p=H
            else:
                H_p=LagrangeExtrapolate(t,H,t[i-1]+(1/2))  
            U=np.real(propagator('EM',H_p,[],dt))
            C_p=np.dot(U,C)
            # update
            P_p=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L)
            H_c=H+(1/2)*(E_0-H)
            U=np.real(propagator('EM',H_c,[],dt))
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
            Hdt=computeE_0(R_I,Z_I,P_new,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L)
            U=np.real(propagator(select,H,Hdt,dt))
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
        case 'AETRS':
            Hdt=LagrangeExtrapolate(t,energies,tnew)
            U=np.real(propagator(select,H,Hdt,dt))
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
        case 'CAETRS':    
            Hdt=LagrangeExtrapolate(t,energies,tnew)
            U=np.real(propagator(select,H,Hdt,dt))
            C_p=np.dot(U,C)
            P_p=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_p[u][v] += C_p[u][p]*C_p[v][p]
                    P_p[u][v] *=2
            E_0=computeE_0(R_I,Z_I,P_p,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L)
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
        case 'CFM4':
            Ht1=LagrangeExtrapolate(t,energies,(t[i-1])+((1/2)-(np.sqrt(3)/6)))
            Ht2=LagrangeExtrapolate(t,energies,(t[i-1])+((1/2)+(np.sqrt(3)/6)))
            U=np.real(propagator(select,Ht1,Ht2,dt))
            C_new=np.dot(U,C)
            P_new=np.array([[0., 0.],[0., 0.]])
            for u in range(0,2) :
                for v in range(0,2) :
                    for p in range(0,1) :
                        P_new[u][v] += C_new[u][p]*C_new[v][p]
                    P_new[u][v] *=2
        case _:
            raise TypeError("Invalid propagator")
    
    return P_new

def rttddft(nsteps,dt,propagator,SCFiterations,L,N_i,alpha,Coef,R_I,Z_I,Cpp,P_init):
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
    print('Contributing authors:\nMatthew Thompson - MatThompson@lincoln.ac.uk\nMatt Watkins - MWatkins@lincoln.ac.uk\n'+
    'Date of last edit: 28/04/2023\n'+
    'Description: A program to perform RT-TDDFT exclusively in Python.\n'+ 
    '\t     A variety of propagators (CN, EM, ETRS, etc.) can be \n'+
    '\t     selected for benchmarking and testing. Comments on \n'+
    '\t     the approaches used are in progress.\n'+
    '\n'+'System requirements:\n'+
    '- Python >= 3.10\n'+
    '- Numpy >= 1.13\n'+
    '- Sympy >=1.7.1\n'+
    '- Scipy >= 0.19\n'+
    '- PySCF >=2.0a, requires either Unix-based system or WSL\n'+
    '--------------------------------------------------------------------\n')

    print('Ground state calculations:\n')
    # Performing calculation of all constant variables to remove the need to repeat it multiple times.
    r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II=dftSetup(R_I,alpha,Coef,L,N_i,Z_I)

    # Compute the ground state first
    P,H,C,KS=computeDFT(R_I,alpha,Coef,L,N_i,P_init,Z_I,Cpp,SCFiterations,r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II)
    energies=[]
    mu=[]
    t=np.arange(0,nsteps*dt,dt)

    for i in range(0,nsteps):
        print('--------------------------------------------------------------------\nPropagation timestep: '+str(i+1))
        #Applying perturbation
        KS=deltaKick(KS,2e-5,[1,0,0],t[i],r_x,r_y,r_z,CGF_He,CGF_H,dr)
        #Getting perterbed density matrix
        P=GetP(KS,S)
        # Propagating
        if i<3 and propagator=='AETRS'or'CAETRS'or'CFM4':
            P=propagate(R_I,Z_I,P,H,C,dt,'ETRS',N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i)
        elif i>2 and propagator=='AETRS'or'CAETRS'or'CFM4':
            P=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,energies[i-1],t,energies,t[i],i)
        else:
            P=propagate(R_I,Z_I,P,H,C,dt,propagator,N_i,Cpp,r_x,r_y,r_z,N,dr,CGF_He,CGF_H,G_u,G_v,G_w,delta_T,E_self,E_II,L,[],t,energies,t[i],i)
        # Converging on accurate KS and P
        P,H,C,KS=computeDFT(R_I,alpha,Coef,L,N_i,P,Z_I,Cpp,SCFiterations,r_x,r_y,r_z,N,dr,GTOs_He,CGF_He,GTOs_H,CGF_H,G_u,G_v,G_w,PW_He_G,PW_H_G,S,delta_T,E_self,E_II)
        # Information Collection
        D_x,D_y,D_z,D_tot=transition_dipole_tensor_calculation(r_x,r_y,r_z,CGF_He,CGF_H,dr)
        mu_t=np.trace(np.dot(D_tot,P))
        energies.append(H)
        mu.append(mu_t)

        print('Total dipole moment: '+str(mu_t))
    
    return energies,mu

def transition_dipole_tensor_calculation(r_x,r_y,r_z,CGF_He,CGF_H,dr) :

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
    f=lagrange(t,H)

    return f(tnew)

def pop_analysis(P,S) :
    PS = np.matmul(P,S)
    pop_total = np.trace(PS)
    pop_He = PS[0,0]
    pop_H = PS[1,1]
    
    return pop_total, pop_He, pop_H

#%%
# DFT parameters
SCFiterations=20

#Grid parameters
L=10.
N_i=60

# Molecule parameters
alpha = [[0.3136497915, 1.158922999, 6.362421394],[0.1688554040, 0.6239137298, 3.425250914]] #alpha[0] for He, alpha[1] for H
Coef = [0.4446345422, 0.5353281423, 0.1543289673] #Coefficients are the same for both He and H
R_I = [np.array([0.,0.,0.]), np.array([0.,0.,1.4632])] #R_I[0] for He, R_I[1] for H.
Z_I = [2.,1.]
Cpp = [[-9.14737128,1.71197792],[-4.19596147,0.73049821]] #He, H
P_init=np.array([[1.333218,0.],[0.,0.666609]])

energies,mu=rttddft(10,0.1,'EM',SCFiterations,L,N_i,alpha,Coef,R_I,Z_I,Cpp,P_init)
# %%
plt.plot(np.arange(0,10*0.1,0.1),energies)
# %%
mu=np.array(mu)
c=299792458
h=6.62607015e-34
sp=scipy.fft.rfft(mu)
freq = scipy.fft.rfftfreq(mu.size,(0.1*2.4188843265857e-17))
freqshift=scipy.fft.fftshift(freq)
ld=c/freq
en=h*freq
plt.plot(ld,np.abs(sp))
plt.xlabel('Wavelength, $\lambda$')
plt.ylabel('Intensity')
#plt.xlim([0, 5e-7])


# %%
