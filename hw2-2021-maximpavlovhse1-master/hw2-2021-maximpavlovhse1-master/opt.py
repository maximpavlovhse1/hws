#!/usr/bin/env python3

from collections import namedtuple


Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
Result.__doc__ = """Результаты оптимизации

Attributes
----------
nfev : int
    Полное число вызовов модельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива — менее nfev
gradnorm : float
    Норма градиента на финальном итерационном шаге
x : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
"""

import numpy as np
from scipy import integrate

def f(z , h_0, w):
    under_int = lambda x: 1/(np.sqrt((1-w)*(1+x)**3 + w))
    int = np.array(([integrate.quad(under_int, 0, _) for _ in z]))
    
    return 5*np.log10((3*(10**11)*(1+z)/h_0)*int[:, 0])-5    
    

        
def j(z, h_0, w):
    under_int = lambda x: 1/np.sqrt((1-w)*(1+x)**3 + w)
    der = lambda x: (((x+1)**3)-1)/(2*(w-(w-1)*(x+1)**3)**1.5)
    
    jac_0 = []
     
    for i in range(len(z)):
        k1 = integrate.quad(der, 0, z[i])[0]
        k2 = integrate.quad(under_int, 0, z[i])[0]
        jac_0.append((5*k1)/(np.log(10)*k2))
        
    jac = np.empty((z.size, 2), dtype=np.float)
    jac[:, 1] = jac_0
    jac[:, 0] = - 5 * 1/(np.log(10)*h_0)
    return jac    
    

def gauss_newton(y, f, j, x0, k=0.1, tol=1e-4):
    x = np.asarray(x0)
    i = 0
    cost = []
    while True:
        i += 1
        r = f(*x) - y
        cost.append(0.5 * np.dot(r,r))
        jac = j(*x)
        g = np.dot(jac.T, r)
        g_norm = np.linalg.norm(g)
        delta_x = np.linalg.solve(jac.T @ jac, -g)
        x[1:] = x[1:] + k*delta_x
        if len(cost) > 2 and np.abs(cost[-1] - cost[-2]) <= tol * cost[-1]:
            break
    return Result(nfev=i, cost=cost, gradnorm=g_norm, x=x)


    
def lm(y, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-4):
    x = np.asarray(x0)
    i = 0
    cost = []
    lmbd = lmbd0
    while True:
        i += 1
        r = y - f(*x)
        cost.append(0.5 * np.dot(r,r))
        jac = j(*x)
        g = np.dot(jac.T, r)
        g_norm = np.linalg.norm(g)
        fi = 0.5 * (np.dot(r,r))
        d_x1 = np.linalg.solve((jac.T @ jac + lmbd * np.eye(2, 2)), jac.T@r)
        d_x2 = np.linalg.solve((jac.T @ jac + lmbd/nu * np.eye(2, 2)), jac.T@r)
        fi_new1 = 0.5*np.dot(r,r) + np.dot(jac.T@r, d_x1) + 0.5*np.dot(jac.T @ jac @ d_x1, d_x1)
        fi_new2 = 0.5*np.dot(r,r) + np.dot(jac.T@r, d_x2) + 0.5*np.dot(jac.T @ jac @ d_x2, d_x2)
        if fi_new2 <= fi:
            lmbd = lmbd/nu
        elif fi_new2 > fi and fi_new2 <= fi_new1:
            lmbd = lmbd
        elif fi_new2 > fi and fi_new2 > fi_new1:
            w = 0
            while fi_new1 > fi:
                lmbd = lmbd*3
                d_x= np.linalg.solve((jac.T @ jac + lmbd * np.eye(2, 2)), jac.T @ r)
                fi_new1 = 0.5*np.dot(r,r) + np.dot(jac.T@r, d_x) + 0.5*np.dot(jac.T @ jac @ d_x, d_x)
                w = w+1
            lmbd = lmbd0*(3**w)
                
            
        x[1:] = x[1:] + np.linalg.solve((jac.T @ jac + lmbd * np.eye(2, 2)), jac.T@r)
        
    
        
        if len(cost) > 2 and np.abs(cost[-1] - cost[-2]) <= tol * cost[-1]:
            break
    return Result(nfev = i, cost = cost,gradnorm=g_norm, x=x)
        
    
                
            
        
        



