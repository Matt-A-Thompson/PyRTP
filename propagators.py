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

def propagate_noKSPC_noPPC(P,KS,KS_prev,select,dt,i,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers):
    match select:
        case 'CN':
            if i==0:
                KS_half=KS
            else:
                KS_half=((1.5)*KS)-(0.5*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            U,functioncalls,timers=propagator(select,KS_half,0,dt,functioncalls,timers)
        case 'EM':
            if i==0:
                KS_half=KS
            else:
                KS_half=((1.5)*KS)-(0.5*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            U,functioncalls,timers=propagator(select,KS_half,0,dt,functioncalls,timers)
        case 'ETRS':
            if i==0:
                KS_half=KS
            else:
                KS_half=((1.5)*KS)-(0.5*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            U,functioncalls,timers=propagator(select,KS_half,0,dt,functioncalls,timers)
            P_next=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
            KS_next,functioncalls,timers=GetKS(P_next,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
            U,functioncalls,timers=propagator(select,KS,KS_next,dt,functioncalls,timers)
        case 'AETRS':
            if i==0:
                KS_next=KS
            else:
                KS_next=((2)*KS)-(1*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            U,functioncalls,timers=propagator(select,KS,KS_next,dt,functioncalls,timers)
        case 'CAETRS':
            raise TypeError("CAETRS is not available without PC algorithm")
        case 'CFM4':
            if i==0:
                KS_t1=KS
                KS_t2=KS
            else:
                KS_t1=((1+(1/2)-(np.sqrt(3)/6))*KS)-(((1/2)-(np.sqrt(3)/6))*KS_prev)
                KS_t2=((1+(1/2)+(np.sqrt(3)/6))*KS)-(((1/2)+(np.sqrt(3)/6))*KS_prev)
            U,functioncalls,timers=propagator(select,KS_t1,KS_t2,dt,functioncalls,timers)
    P_new=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
    KS_new,functioncalls,timers=GetKS(P_new,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
    return P_new,KS_new,timers

def propagate_KSPC_noPPC(P,KS,KS_prev,select,dt,i,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers):
    match select:
        case 'CN':
            if i==0:
                KS_half_p=KS
            else:
                KS_half_p=((1.5)*KS)-(0.5*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            for i in range(0,100):
                U,functioncalls,timers=propagator(select,KS_half_p,0,dt,functioncalls,timers)
                P_next=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                P_half=(P_next+P)/2
                KS_half_c,functioncalls,timers=GetKS(P_half,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
                if np.all(abs(KS_half_p-KS_half_c)<=1e-6)==True:
                    break
                else:
                    KS_half_p=KS_half_c
            U,functioncalls,timers=propagator(select,KS_half_c,0,dt,functioncalls,timers)
        case 'EM':
            if i==0:
                KS_half_p=KS
            else:
                KS_half_p=((1.5)*KS)-(0.5*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            for i in range(0,100):
                U,functioncalls,timers=propagator(select,KS_half_p,0,dt,functioncalls,timers)
                P_next=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                P_half=(P_next+P)/2
                KS_half_c,functioncalls,timers=GetKS(P_half,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
                if np.all(abs(KS_half_p-KS_half_c)<=1e-6)==True:
                    break
                else:
                    KS_half_p=KS_half_c
            U,functioncalls,timers=propagator(select,KS_half_c,0,dt,functioncalls,timers)
        case 'ETRS':
            if i==0:
                KS_half_p=KS
            else:
                KS_half_p=((1.5)*KS)-(0.5*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            for i in range(0,100):
                U,functioncalls,timers=propagator(select,KS_half_p,0,dt,functioncalls,timers)
                P_next=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                P_half=(P_next+P)/2
                KS_half_c,functioncalls,timers=GetKS(P_half,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
                if np.all(abs(KS_half_p-KS_half_c)<=1e-6)==True:
                    break
                else:
                    KS_half_p=KS_half_c

            U,functioncalls,timers=propagator(select,KS_half_c,0,dt,functioncalls,timers)
            P_next=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
            KS_next,functioncalls,timers=GetKS(P_next,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
            U,functioncalls,timers=propagator(select,KS,KS_next,dt,functioncalls,timers)
            
        case 'AETRS':
            if i==0:
                print('Note: AETRS with PC algorithm is referred to as CAETRS, which can be selected.')
                KS_next_p=KS
            else:
                KS_next_p=((2)*KS)-(1*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            for i in range(0,100):
                U,functioncalls,timers=propagator(select,KS,KS_next_p,dt,functioncalls,timers)
                P_next=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                KS_next_c,functioncalls,timers=GetKS(P_next,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
                if np.all(abs(KS_next_p-KS_next_c)<=1e-6)==True:
                    break
                else:
                     KS_next_p=KS_next_c
            U,functioncalls,timers=propagator(select,KS,KS_next_c,dt,functioncalls,timers)
        case 'CAETRS':
            if i==0:
                KS_next_p=KS
            else:
                KS_next_p=((2)*KS)-(1*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            for i in range(0,100):
                U,functioncalls,timers=propagator(select,KS,KS_next_p,dt,functioncalls,timers)
                P_next=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                KS_next_c,functioncalls,timers=GetKS(P_next,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
                if np.all(abs(KS_next_p-KS_next_c)<=1e-6)==True:
                    break
                else:
                     KS_next_p=KS_next_c
            U,functioncalls,timers=propagator(select,KS,KS_next_c,dt,functioncalls,timers)
        case 'CFM4':
            if i==0:
                print('Warning: CFM4 does not have PC algorithm, running without...')
                KS_t1=KS
                KS_t2=KS
            else:
                KS_t1=((1+(1/2)-(np.sqrt(3)/6))*KS)-(((1/2)-(np.sqrt(3)/6))*KS_prev)
                KS_t2=((1+(1/2)+(np.sqrt(3)/6))*KS)-(((1/2)+(np.sqrt(3)/6))*KS_prev)
            U,functioncalls,timers=propagator(select,KS_t1,KS_t2,dt,functioncalls,timers)

    P_new=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
    KS_new,functioncalls,timers=GetKS(P_new,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
    return P_new,KS_new,timers

def propagate_noKSPC_PPC(P,KS,KS_prev,select,dt,i,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers):
    match select:
        case 'CN':
            if i==0:
                KS_half=KS
            else:
                KS_half=((1.5)*KS)-(0.5*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            U,functioncalls,timers=propagator(select,KS_half,0,dt,functioncalls,timers)
            P_new_p=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
            KS_new_p,functioncalls,timers=GetKS(P_new_p,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
            for i in range(0,100):
                U=functioncalls,timers=propagator(select,((KS_new_p+KS)/2),0,dt,functioncalls,timers)
                P_new_c=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                if np.all(abs(P_new_p-P_new_c)<=1e-6)==True:
                    break
                else:
                    P_new_p=P_new_c
            P_new=P_new_c
        case 'EM':
            if i==0:
                KS_half=KS
            else:
                KS_half=((1.5)*KS)-(0.5*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            U,functioncalls,timers=propagator(select,KS_half,0,dt,functioncalls,timers)
            P_new_p=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
            KS_new_p,functioncalls,timers=GetKS(P_new_p,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
            for i in range(0,100):
                U=functioncalls,timers=propagator(select,((KS_new_p+KS)/2),0,dt,functioncalls,timers)
                P_new_c=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                if np.all(abs(P_new_p-P_new_c)<=1e-6)==True:
                    break
                else:
                    P_new_p=P_new_c
            P_new=P_new_c
        case 'ETRS':
            if i==0:
                KS_half=KS
            else:
                KS_half=((1.5)*KS)-(0.5*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            U,functioncalls,timers=propagator(select,KS_half,0,dt,functioncalls,timers)
            P_next=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
            KS_next,functioncalls,timers=GetKS(P_next,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
            U,functioncalls,timers=propagator(select,KS,KS_next,dt,functioncalls,timers)
            P_new_p=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
            KS_new_p,functioncalls,timers=GetKS(P_new_p,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
            for i in range(0,100):
                U=functioncalls,timers=propagator(select,((KS_new_p+KS)/2),0,dt,functioncalls,timers)
                P_new_c=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                if np.all(abs(P_new_p-P_new_c)<=1e-6)==True:
                    break
                else:
                    P_new_p=P_new_c
            P_new=P_new_c
        case 'AETRS':
            if i==0:
                KS_next=KS
            else:
                KS_next=((2)*KS)-(1*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            U,functioncalls,timers=propagator(select,KS,KS_next,dt,functioncalls,timers)
            P_new_p=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
            KS_new_p,functioncalls,timers=GetKS(P_new_p,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
            for i in range(0,100):
                U=functioncalls,timers=propagator(select,((KS_new_p+KS)/2),0,dt,functioncalls,timers)
                P_new_c=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                if np.all(abs(P_new_p-P_new_c)<=1e-6)==True:
                    break
                else:
                    P_new_p=P_new_c
            P_new=P_new_c
        case 'CAETRS':
            raise TypeError("CAETRS is not available without PC algorithm")
        case 'CFM4':
            if i==0:
                KS_t1=KS
                KS_t2=KS
            else:
                KS_t1=((1+(1/2)-(np.sqrt(3)/6))*KS)-(((1/2)-(np.sqrt(3)/6))*KS_prev)
                KS_t2=((1+(1/2)+(np.sqrt(3)/6))*KS)-(((1/2)+(np.sqrt(3)/6))*KS_prev)
            U,functioncalls,timers=propagator(select,KS_t1,KS_t2,dt,functioncalls,timers)
            P_new_p=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
            KS_new_p,functioncalls,timers=GetKS(P_new_p,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
            for i in range(0,100):
                U=functioncalls,timers=propagator(select,((KS_new_p+KS)/2),0,dt,functioncalls,timers)
                P_new_c=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                if np.all(abs(P_new_p-P_new_c)<=1e-6)==True:
                    break
                else:
                    P_new_p=P_new_c
            P_new=P_new_c

    KS_new,functioncalls,timers=GetKS(P_new,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
    return P_new,KS_new,timers

def propagate_KSPC_PPC(P,KS,KS_prev,select,dt,i,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers):
    match select:
        case 'CN':
            if i==0:
                KS_half_p=KS
            else:
                KS_half_p=((1.5)*KS)-(0.5*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            for i in range(0,100):
                U,functioncalls,timers=propagator(select,KS_half_p,0,dt,functioncalls,timers)
                P_next=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                P_half=(P_next+P)/2
                KS_half_c,functioncalls,timers=GetKS(P_half,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
                if np.all(abs(KS_half_p-KS_half_c)<=1e-6)==True:
                    break
                else:
                    KS_half_p=KS_half_c
            U,functioncalls,timers=propagator(select,KS_half_c,0,dt,functioncalls,timers)
            P_new_p=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
            KS_new_p,functioncalls,timers=GetKS(P_new_p,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
            for i in range(0,100):
                U=functioncalls,timers=propagator(select,((KS_new_p+KS)/2),0,dt,functioncalls,timers)
                P_new_c=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                if np.all(abs(P_new_p-P_new_c)<=1e-6)==True:
                    break
                else:
                    P_new_p=P_new_c
            P_new=P_new_c
        case 'EM':
            if i==0:
                KS_half_p=KS
            else:
                KS_half_p=((1.5)*KS)-(0.5*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            for i in range(0,100):
                U,functioncalls,timers=propagator(select,KS_half_p,0,dt,functioncalls,timers)
                P_next=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                P_half=(P_next+P)/2
                KS_half_c,functioncalls,timers=GetKS(P_half,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
                if np.all(abs(KS_half_p-KS_half_c)<=1e-6)==True:
                    break
                else:
                    KS_half_p=KS_half_c
            U,functioncalls,timers=propagator(select,KS_half_c,0,dt,functioncalls,timers)
            P_new_p=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
            KS_new_p,functioncalls,timers=GetKS(P_new_p,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
            for i in range(0,100):
                U=functioncalls,timers=propagator(select,((KS_new_p+KS)/2),0,dt,functioncalls,timers)
                P_new_c=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                if np.all(abs(P_new_p-P_new_c)<=1e-6)==True:
                    break
                else:
                    P_new_p=P_new_c
            P_new=P_new_c
        case 'ETRS':
            if i==0:
                KS_half_p=KS
            else:
                KS_half_p=((1.5)*KS)-(0.5*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            for i in range(0,100):
                U,functioncalls,timers=propagator(select,KS_half_p,0,dt,functioncalls,timers)
                P_next=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                P_half=(P_next+P)/2
                KS_half_c,functioncalls,timers=GetKS(P_half,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
                if np.all(abs(KS_half_p-KS_half_c)<=1e-6)==True:
                    break
                else:
                    KS_half_p=KS_half_c

            U,functioncalls,timers=propagator(select,KS_half_c,0,dt,functioncalls,timers)
            P_next=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
            KS_next,functioncalls,timers=GetKS(P_next,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
            U,functioncalls,timers=propagator(select,KS,KS_next,dt,functioncalls,timers)
            P_new_p=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
            KS_new_p,functioncalls,timers=GetKS(P_new_p,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
            for i in range(0,100):
                U=functioncalls,timers=propagator(select,((KS_new_p+KS)/2),0,dt,functioncalls,timers)
                P_new_c=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                if np.all(abs(P_new_p-P_new_c)<=1e-6)==True:
                    break
                else:
                    P_new_p=P_new_c
            P_new=P_new_c
        case 'AETRS':
            if i==0:
                print('Note: AETRS with PC algorithm is referred to as CAETRS, which can be selected.')
                KS_next_p=KS
            else:
                KS_next_p=((2)*KS)-(1*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            for i in range(0,100):
                U,functioncalls,timers=propagator(select,KS,KS_next_p,dt,functioncalls,timers)
                P_next=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                KS_next_c,functioncalls,timers=GetKS(P_next,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
                if np.all(abs(KS_next_p-KS_next_c)<=1e-6)==True:
                    break
                else:
                     KS_next_p=KS_next_c
            U,functioncalls,timers=propagator(select,KS,KS_next_c,dt,functioncalls,timers)
            P_new_p=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
            KS_new_p,functioncalls,timers=GetKS(P_new_p,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
            for i in range(0,100):
                U=functioncalls,timers=propagator(select,((KS_new_p+KS)/2),0,dt,functioncalls,timers)
                P_new_c=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                if np.all(abs(P_new_p-P_new_c)<=1e-6)==True:
                    break
                else:
                    P_new_p=P_new_c
            P_new=P_new_c
        case 'CAETRS':
            if i==0:
                KS_next_p=KS
            else:
                KS_next_p=((2)*KS)-(1*KS_prev) # might need changing (paper uses KS_{N+1/2}=2KS_{N}-KS_{N-1/2})
            for i in range(0,100):
                U,functioncalls,timers=propagator(select,KS,KS_next_p,dt,functioncalls,timers)
                P_next=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                KS_next_c,functioncalls,timers=GetKS(P_next,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
                if np.all(abs(KS_next_p-KS_next_c)<=1e-6)==True:
                    break
                else:
                     KS_next_p=KS_next_c
            U,functioncalls,timers=propagator(select,KS,KS_next_c,dt,functioncalls,timers)
            P_new_p=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
            KS_new_p,functioncalls,timers=GetKS(P_new_p,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
            for i in range(0,100):
                U=functioncalls,timers=propagator(select,((KS_new_p+KS)/2),0,dt,functioncalls,timers)
                P_new_c=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                if np.all(abs(P_new_p-P_new_c)<=1e-6)==True:
                    break
                else:
                    P_new_p=P_new_c
            P_new=P_new_c
        case 'CFM4':
            if i==0:
                print('Warning: CFM4 does not have PC algorithm, running without...')
                KS_t1=KS
                KS_t2=KS
            else:
                KS_t1=((1+(1/2)-(np.sqrt(3)/6))*KS)-(((1/2)-(np.sqrt(3)/6))*KS_prev)
                KS_t2=((1+(1/2)+(np.sqrt(3)/6))*KS)-(((1/2)+(np.sqrt(3)/6))*KS_prev)
            U,functioncalls,timers=propagator(select,KS_t1,KS_t2,dt,functioncalls,timers)
            P_new_p=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
            KS_new_p,functioncalls,timers=GetKS(P_new_p,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
            for i in range(0,100):
                U=functioncalls,timers=propagator(select,((KS_new_p+KS)/2),0,dt,functioncalls,timers)
                P_new_c=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
                if np.all(abs(P_new_p-P_new_c)<=1e-6)==True:
                    break
                else:
                    P_new_p=P_new_c
            P_new=P_new_c

    P_new=np.dot(U,np.dot(P,np.conj(np.transpose(U))))
    KS_new,functioncalls,timers=GetKS(P_new,phi,N,N_i,dr,Z_I,r_x,r_y,r_z,R_I,G_u,G_v,G_w,L,delta_T,Cpp,alpha_PP,V_NL,functioncalls,timers)
    return P_new,KS_new,timers