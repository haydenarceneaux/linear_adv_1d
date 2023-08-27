"""
Roe-Riemann Solver for the 1D linear advection equation
rec = 1 - Upwind (1st Order)
rec = 2 - MUSCL (2nd Order)
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from numba import njit, jit, prange

start = time.time()

rec = 1
 # reconstruction scheme
N = int(20001) # cell count
cfl = 0.1 # cfl number < 1.0

tf = 0.5
a = 1.0

x0 = -1
x1 = 1
x0 = -1
x1 = 1
#N = int(101)
x = np.linspace(x0,x1,num=N)
dx = x[1]-x[0]

U = np.zeros(N)
U = (x>=0).astype(float)
#for i in range(0,N):
#    U[i] = np.exp(-8*x[i]**2)
    
@njit
def Roe_flux(UR,UL,a):
    fip12 = 0.5*a*(UL+UR)-0.5*a*(UR-UL)
    return fip12

@njit
def Upwind(U,i):
    URip12 = U[i+1]
    ULip12 = U[i]
    return URip12, ULip12

@njit
def MUSCL(U,i):
    URip12 = U[i+1]-phi(r_fun(U,i+1))/2*(U[i+2]-U[i+1])
    ULip12 = U[i]+phi(r_fun(U,i))/2*(U[i+1]-U[i])
    return URip12, ULip12

@njit
def MUSCL3(U,i):
    k = 1/3
    #URip12 = U[i+1]-phi(r_fun(U,i+1))/2*(U[i+2]-2*U[i+1]-2*U[i])/3
    #ULip12 = U[i]+phi(r_fun(U,i))/2*(U[i+1]+U[i])
    
    URip12 = U[i+1]-phi(r_fun(U,i+1))/4*((1+k)*(U[i+1]-U[i])+(1-k)*(U[i+2]-U[i+1]))
    ULip12 = U[i]+phi(r_fun(U,i))/4*((1+k)*(U[i+1]-U[i])-(1-k)*(U[i]-U[i-1]))
    #URip12 = Uip1-phiip1/4.*((1+kappa)*(Uip1-Ui)+(1-kappa)*(Uip2-Uip1));
    #ULip12 = Ui+phiip1/4.*((1-kappa)*(Ui-Uim1)+(1+kappa)*(Uip1-Ui));

    return URip12, ULip12
@njit
def r_fun(U,i):
    r = (U[i]-U[i-1])/((U[i+1]-U[i])+1e-6)
    return r

@njit
def phi(r):
    #phi = max(0,min(1,r)) # minmod
    phi = (r+abs(r))/(1+abs(r)) # van Leer
    #phi = 1.5*(r**2+r)/(r**2+r+1) # ospre
    #phi = max(0,min(2*r,1),min(r,2)) #superbee
    #phi = 0.5*(r+1)*min(min(1,4*r/(r+1)),min(1,4/(r+1)))
    #phi = 07401
    return phi

@njit
def k_exact2(U,i):
    # kinda works
    UR = U[i+1]-phi(r_fun(U,i+1))*(0.5*(U[i+1]-U[i])+1/8*(U[i+2]-2*U[i+1]+U[i]))
    UL = U[i]+phi(r_fun(U,i))*(0.5*(U[i]-U[i-1])+1/8*(U[i+1]-2*U[i]+U[i-1]))
    
    
    #UR = U[i+1]-0.5*(U[i+2]-U[i+1])+1/8*(U[i+2]-2*U[i+1]+U[i])
    #UL = U[i]-0.5*(U[i+1]-U[i])+1/8*(U[i+1]-2*U[i]+U[i-1])
    return UR, UL

@njit
def k_exact3(U,i):
    #works!!!
    #UR = U[i+1]-phi(r_fun(U,i+1))*(0.5*(U[i+2]-U[i+1])+1/8*(U[i+2]-2*U[i+1]+U[i])-1/48*(2*U[i]-5*U[i+1]+4*U[i+2]-1*U[i+3]))
    #UL = U[i]+phi(r_fun(U,i))*(0.5*(U[i+1]-U[i])+1/8*(U[i+1]-2*U[i]+U[i-1])+1/48*(2*U[i-1]-5*U[i]+4*U[i+1]-1*U[i+2]))
    
    UR = U[i+1]-phi(r_fun(U,i+1))*(0.5*(U[i+2]-U[i+1])+1/8*(U[i+2]-2*U[i+1]+U[i])-1/48*(2*U[i-1]-5*U[i]+4*U[i+1]-1*U[i+2]))
    UL = U[i]+phi(r_fun(U,i))*(0.5*(U[i+1]-U[i])+1/8*(U[i+1]-2*U[i]+U[i-1])+1/48*(2*U[i]-5*U[i-1]+4*U[i]-1*U[i+1]))
    
    return UR, UL

@njit
def order(rec,U,i):
    if rec == 1:
        URip12, ULip12 = Upwind(U,i)
        URim12, ULim12 = Upwind(U,i-1)
    elif rec == 2:
        URip12, ULip12 = MUSCL(U,i)
        URim12, ULim12 = MUSCL(U,i-1)
    elif rec == 3:
        URip12, ULip12 = MUSCL3(U,i)
        URim12, ULim12 = MUSCL3(U,i-1)
    elif rec == 4:
        URip12, ULip12 = k_exact2(U,i)
        URim12, ULim12 = k_exact2(U,i-1)
    elif rec == 5:
        URip12, ULip12 = k_exact3(U,i)
        URim12, ULim12 = k_exact3(U,i-1)
    return URip12, ULip12, URim12, ULim12


@njit(parallel=False,fastmath=True)
def time_loop(U,tf,rec,dx,a,cfl,N):
    Iter = 0.0
    t = 0.0
    fip12 = np.zeros(N)
    fim12 = np.zeros(N)
    Unp = np.zeros(N)
    while(t<tf):
        dt = dx/a*cfl
        t += dt
        Iter += 1
        for i in prange(0,N):
            if i < 2:
                fip12[i] = 0
                fim12[i] = 0
            elif i >= N-3:
                fip12[i] = 0
                fim12[i] = 0
            else:
                URip12, ULip12, URim12, ULim12 = order(rec,U,i)
                fip12[i] = Roe_flux(URip12,ULip12,a)
                fim12[i] = Roe_flux(URim12,ULim12,a)
                
            Unp[i] = U[i]+dt*a/dx*(fim12[i]-fip12[i])
        
        U = Unp
        
    return t,Iter,U

#t,Iter,U=time_loop(U,tf,rec,dx,a,cfl,N)
    
t,Iter,U = time_loop(U,tf,rec,dx,a,cfl,N)

plt.plot(x,U,'k')
#plt.axis([0.4,0.55,-0.2,1.2])
plt.xlabel("x")
plt.ylabel("q")
plt.title("Linear Advection Equation")
plt.axis([0,1,-0.2,1.2])

print("time elapsed: " + str(round((time.time() - start),1)) + " seconds.")