import numpy as np
from scipy import  pi, exp, real, log
from scipy.integrate import quad, quadrature, trapz
from scipy.optimize import fminbound
from scipy.stats import norm
import random
import math

class Heston(object):

    #Parameters of the class
    def __init__(self,S=1,r=0.025,q=0.0,kappa=2,vLong=0.05,sigma=0.3,v0=0.02,rho=-0.7):
        self.S = S 
        self.r = r
        self.q = q
        self.kappa = kappa
        self.vLong = vLong
        self.sigma = sigma
        self.v0 = v0
        self.rho = rho
        
        ############################# Heston Call/Put #####################################
    
    def heston_char_fkt_1(T,r,q,u,v0,vLong,kappa,sigma,rho):
        gamma = kappa - 1j*rho*sigma*u
        d = np.sqrt( gamma**2 + (sigma**2)*u*(u+1j) )
        g = (gamma - d)/(gamma + d)
        C = (kappa*vLong)/(sigma**2)*((gamma-d)*T-2*np.log((1 - g*exp(-d*T))/( 1 - g ) ))
        D = (gamma - d)/(sigma**2)*((1 - np.exp(-d*T))/
          (1 - g*np.exp(-d*T)))
        return exp(C + D*v0)
    

    @staticmethod
    def heston_f1(u,logeps,v0,T):
        return np.abs(-0.5 * v0 * T * u * u - np.log(u) - logeps)
    
    @staticmethod    
    def heston_f2(u,logeps,v0,vLong,kappa,sigma,rho,T):
        Cinf = (v0+kappa*vLong*T)/sigma*np.sqrt(1-rho**2)
        return np.abs(-Cinf * u - np.log(u) - logeps) 


    def heston_call_piterbarg(self,K,T):
        a = (self.v0 * T)**0.5 
        d1 = (log(self.S /K) + ((self.r-self.q) + self.v0 / 2) * T) / a
        d2 = d1 - a
        BSCall = self.S * exp(-self.q*T) * norm.cdf(d1) - K * exp(-self.r*T) * norm.cdf(d2)
        logeps = log(0.00001)
        F = self.S*exp((self.r-self.q)*T)
        x = log(K/F)
        umax1 = fminbound(Heston.heston_f1,0,1000,args=(logeps,self.v0,T))
        umax2 = fminbound(Heston.heston_f2,0,1000,args=(logeps,self.v0,self.vLong,self.kappa,self.sigma,self.rho, T,))
        umax = max(umax1,umax2)
        X = np.linspace(0,umax,1000)
        integrand = lambda k: real(exp(-1j*k*x)/(k**2 + 0.25) *(exp(-0.5*T*self.v0*(k**2 + 0.25))-
                             Heston.heston_char_fkt_1(T,self.r,self.q,k - 0.5*1j,self.v0,self.vLong,self.kappa,self.sigma,self.rho)))
        integral = trapz(integrand(X),x=X)
        return (BSCall + np.sqrt(F*K)/pi * np.exp(-self.r*T) * integral)                                                                     

    def heston_put_piterbarg(self, K, T): 
        return Heston.heston_call_piterbarg(self, K, T) - self.S * np.exp(-self.q * T) + K * np.exp(-self.r * T)
    
    def heston_call_piterbarg_delta(self,K,T):
        a = (self.v0 * T)**0.5 
        d1 = (log(self.S /K) + ((self.r-self.q) + self.v0 / 2) * T) / a

        BSCalldelta = exp(-self.q*T) * norm.cdf(d1)
        logeps = log(0.00001)
        F = self.S*exp((self.r-self.q)*T)
        x = log(K/F)
        umax1 = fminbound(Heston.heston_f1,0,1000,args=(logeps,self.v0,T))
        umax2 = fminbound(Heston.heston_f2,0,1000,args=(logeps,self.v0,self.vLong,self.kappa,self.sigma,self.rho, T,))
        umax = max(umax1,umax2)
        X = np.linspace(0,umax,1000)
        integrand1 = lambda k: real(exp(-1j*k*x)/(k**2 + 0.25) *(exp(-0.5*T*self.v0*(k**2 + 0.25))-
                             Heston.heston_char_fkt_1(T,self.r,self.q,k - 0.5*1j,self.v0,self.vLong,self.kappa,self.sigma,self.rho)))
        integrand2 = lambda k: real((1j*k)/F * exp(-1j*k*x)/(k**2 + 0.25) *(exp(-0.5*T*self.v0*(k**2 + 0.25))-
                             Heston.heston_char_fkt_1(T,self.r,self.q,k - 0.5*1j,self.v0,self.vLong,self.kappa,self.sigma,self.rho)))       
        integral1 = trapz(integrand1(X),x=X)
        integral2 = trapz(integrand2(X),x=X)
        return BSCalldelta + 1/(4 * pi) * K/np.sqrt(F*K) * np.exp(-self.r*T) * integral1 + np.sqrt(F*K)/pi * np.exp(-self.r*T) * integral2
 
    def heston_put_piterbarg_delta(self, K, T): 
        return Heston.heston_call_piterbarg_delta(self, K, T) - np.exp(-self.q * T)
    


