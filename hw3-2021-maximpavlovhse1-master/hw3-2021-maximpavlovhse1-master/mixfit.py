# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 21:13:45 2021

@author: I
"""
from scipy.optimize import minimize
import numpy as np

tau = 0.6
mu1= -2.
mu2 = 0.5
sigma1 = 0.5
sigma2 = 0.6
N = 1000

x1 = np.random.normal(mu1, sigma1, size=(int(N*tau)))
x2 = np.random.normal(mu2, sigma2, size=(int(N*(1-tau))))
x = np.concatenate([x1, x2])

theta = (0.5, np.mean(x)-np.std(x), np.mean(x)+np.std(x), np.var(x), np.var(x))

def max_likelihood(x, tau, mu1, mu2, sigma1, sigma2):
    
    x1 = np.random.normal(mu1, sigma1, size=(int(N*tau)))
    x2 = np.random.normal(mu2, sigma2, size=(int(N*(1-tau))))
    x = np.concatenate([x1, x2])

    theta = (0.5, np.mean(x)-np.std(x), np.mean(x)+np.std(x), np.var(x), np.var(x))
    
    
    
    W_1 = tau/(np.sqrt(2*np.pi)*sigma1)*np.exp(-(x-mu1)**2/(2*sigma1**2))
    W_2 = (1-tau)/(np.sqrt(2*np.pi)*sigma2)*np.exp(-(x-mu2)**2/(2*sigma2**2))
    W = tau/(np.sqrt(2*np.pi)*sigma1)*np.exp(-(x-mu1)**2/(2*sigma1**2)) + (1-tau)/(np.sqrt(2*np.pi)*sigma2)*np.exp(-(x-mu2)**2/(2*sigma2**2))
    der_mu1 = np.sum(W_1*(x-mu1)/(W*sigma1**2))
    der_mu2 = np.sum(W_2*(x-mu2)/(W*sigma2**2))
    der_tau = np.sum((W_1/tau-W_2/(1-tau))/W)
    der_sigma1 = sigma2**2*tau*np.exp((x-mu2)**2/(2*sigma2**2))*(sigma1**2-x**2+2*mu1*x-mu1**2)/(sigma1**3*((tau-1)*sigma1*np.exp((x-mu1)**2/(2*sigma1**2))-sigma2**2*tau*np.exp((x-mu2)**2/(2*sigma2**2))))
    der_sigma2 = sigma1*(tau-1)*np.exp((x-mu1)**2/(2*sigma1**2))*(2*sigma2**2-x**2+2*mu2*x-mu2**2)/(sigma2**3*(tau*sigma2**2*np.exp((x-mu2)**2/(2*sigma2**2))+ (sigma1-sigma1*tau)*np.exp((x-mu1)**2/(2*sigma1**2))))
    #находим градиенты по каждому из параметров, нужно максимизировать
    
    return(minimize(-der_mu1, 0.5), minimize(-der_mu2, np.mean(x)-np.std(x)),
           minimize(-der_tau, np.mean(x)+np.std(x)), minimize(der_sigma1, np.var(x)), minimize(der_sigma2, np.var(x)))
 


    
def em_double_gauss(x, tau, mu1, mu2, sigma1, sigma2, rtol=1e-3):
         W1 = (tau/np.sqrt(2*np.pi*sigma1**2)
          *np.exp(-0.5*(x-mu1)**2 / sigma1**2))
         W2 = ((1-tau)/np.sqrt(2*np.pi*sigma2**2)
          *np.exp(-0.5*(x-mu2)**2 / sigma2**2))
         W = W1+W2
         W1 = np.divide(W1, W,out = np.full_like(W1, 0.5), where = W!=0)
         W2 = np.divide(W2, W,out = np.full_like(W1, 0.5), where = W!=0)
         
         W1, W2 = em_double_gauss(x, *theta)
         mu1 = np.sum(W1*x)/np.sum(W1)
         mu2 = np.sum(W2*x)/np.sum(W2)
         tau = np.sum(W1)/np.sum(W)
         sigma1 = np.sqrt(np.sum(W1 * (x-mu1)**2) / np.sum(W1)) 
         sigma2 = np.sqrt(np.sum(W2 * (x-mu2)**2) / np.sum(W2))
         return(tau, mu1, sigma1, mu2, sigma2)
#rtol можно учесть уже когда мы задали начальное приближение и ищем значения

for i in range(1000):
    if abs(0.5-em_double_gauss(x, *theta)) <= 1e-3:
        break
    theta = em_double_gauss(x, *theta)
    
    

     
    
    
         