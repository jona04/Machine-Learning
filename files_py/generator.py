import random

# StringIO behaves like a file object 
from io import StringIO 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf

import copy

import joblib

class CqkProblem:
    def __init__(self, r, n, d, a, b, low, up):
        self.n = n
        self.r = r
        self.d = list(d)
        self.a = list(a)
        self.b = list(b)
        self.low = list(low)
        self.up = list(up)


def generate_cqk_problem(n):
    d = []
    low = []
    up = []
    b = []
    a = []
    temp = 0
    lb = 0.0
    ub = 0.0
    lower = 10
    upper = 25
    r = 0

    for i in range(n):
        
        b.append(10 + 14*random.random())
        low.append(1 + 14*random.random())
        up.append(1 + 14*random.random())
        if low[i] > up[i]:
            temp = low[i]
            low[i] = up[i]
            up[i] = temp
        
        lb = lb + b[i]*low[i];
        ub = ub + b[i]*up[i];
        
        #Uncorrelated
        d.append(random.randint(10,25))
        a.append(random.randint(10,25))
        
    r = lb + (ub - lb)*0.7;
    
    return CqkProblem( r, n, d, a, b, low, up)


def initial_lambda(p, lamb):
    s0=0.0
    q0=0.0
    slopes = []
    for i in range(p.n):
        slopes.append((p.b[i]/p.d[i])*p.b[i])
        s0 = s0 + (p.a[i] * p.b[i]) / p.d[i]
        q0 = q0 + (p.b[i] * p.b[i]) / p.d[i]
    lamb = (p.r-s0)/q0
    return lamb, slopes


def phi_lambda(p,lamb,phi,deriv,slopes,r):
    deriv = 0.0
    phi = r * -1
    x = []
    
    for i in range(p.n):
        
        x.append( (p.b[i] * lamb + p.a[i])/p.d[i])

        if x[i] < p.low[i]:
            x[i] = p.low[i]
        elif x[i] > p.up[i]:
            x[i] = p.up[i]
        else:
            deriv = deriv + slopes[i];
        phi = phi + p.b[i] * x[i];
    return deriv, phi, x


MAX_IT = 20
INFINITO_NEGATIVO = -999999999;
INFINITO_POSITIVO = 999999999;

def newton(p):
    lambs = [] 
    phis = []
    derivs = []
    phi = 0
    lamb = 0
    alfa = INFINITO_NEGATIVO;
    beta = INFINITO_POSITIVO;
    phi_alfa = 0.0;
    phi_beta = 0.0;
    deriv = 0
    x = []
    r = p.r
    
    lamb, slopes = initial_lambda(p,lamb)
    deriv, phi, x = phi_lambda(p,lamb,phi,deriv,slopes,r)
    lambs.append(lamb)
    derivs.append(deriv)
    phis.append(phi)
    it = 1
#     print(it, deriv, phi,lamb)
    negativo = False
    while phi != 0.0 and it <= MAX_IT:
        if phi > 0:
#             print("positivo")
            beta = lamb
            lambda_n = 0.0
            if deriv > 0.0:
                
                lambda_n = lamb - (phi/deriv)
                if abs(lambda_n - lamb) <= 0.00000000001:
                    phi = 0.0
                    break
                if lambda_n > alfa:
                    lamb = lambda_n
                else:
#                     print("aqui")
                    phi_beta = phi;
#                     lamb = secant(p,x,alfa,beta,phi_alfa,phi_beta,r);
#             if deriv == 0.0:
#                 lamb = breakpoint_to_the_left(p,lamb);
#                 if lamb <= INFINITO_NEGATIVO or lamb >= INFINITO_POSITIVO:
#                     break
                
        else:
            if it == 1:
                negativo = True
#             print("negativo")
            alfa = lamb;
            lambda_n = 0.0;

            if deriv > 0.0:
                lambda_n = lamb - (phi/deriv)
                if abs(lambda_n - lamb) <= 0.00000000001:
                    phi = 0.0
                    break
                
                if lambda_n < beta:
                    lamb = lambda_n
                else:
#                     print("aqui")
                    phi_alfa = phi;
#                     lamb = secant(p,x,alfa,beta,phi_alfa,phi_beta,r);
#             if deriv == 0.0:
#                 lamb = breakpoint_to_the_right(p,lamb)
#                 if lamb <= INFINITO_NEGATIVO or lamb >= INFINITO_POSITIVO:
#                     break
        
        
        deriv, phi, x = phi_lambda(p,lamb,phi,deriv,slopes,r)
        it = it + 1
        lambs.append(lamb)
        derivs.append(deriv)
        phis.append(phi)
        
    if phi == 0.0:
        return it,lambs, derivs, phis,slopes
    elif alfa == beta:
        return -1,lambs, derivs, phis,slopes
    else:
        return -2,lambs, derivs, phis,slopes


lista = []
for i in range(20000):
    n = 10000
    p = generate_cqk_problem(n)
    it,lambs, derivs, phis,slopes = newton(p)
    soma_a = 0
    soma_b = 0
    soma_low = 0
    soma_d = 0
    soma_up = 0
    for i in range(n):
        soma_a += p.a[i]
        soma_b += p.b[i]
        soma_low += p.low[i]
        soma_d += p.d[i]
        soma_up += p.up[i]
    soma_a = soma_a/n
    soma_b = soma_b/n
    soma_low = soma_low/n
    soma_d = soma_d/n
    soma_up = soma_up/n
    r = p.r/n
    
    new_deriv = []
    new_phi = []
    new_lamb = []
    add_lamb = 0.5
    for i in range(4):
#         newLamb = lambs[i+1]+random.uniform(-0.1, 0.1)
        newLamb = lambs[0]+add_lamb
        deriv, phi, x = phi_lambda(p,newLamb,0,0,slopes,p.r)
        new_deriv.append(deriv)
        new_phi.append(phi)
        new_lamb.append(newLamb)
        add_lamb += 0.5
        
    sum_slopes = sum(slopes)
    if it > 3:
        l_rs = [lambs[0],phis[0],derivs[0],new_lamb[0],new_phi[0],new_deriv[0],new_lamb[1],new_phi[1],new_deriv[1],new_lamb[2],new_phi[2],new_deriv[2],new_lamb[3],new_phi[3],new_deriv[3],lambs[-1]]


#         l_rs = [soma_a, soma_b, soma_d, r,lambs[0],phis[0],derivs[0],sum_slopes,lambs[1],phis[1],derivs[1],lambs[2],lambs[3],lambs[-1]]
#         l_rs = [soma_a, soma_b, soma_d, r,lambs[0],lambs[1],lambs[2],lambs[3],lambs[-1]]


        lista.append(l_rs)

np.savetxt('instance_test10k_20k.txt', lista, delimiter = ' ',newline='\n', fmt="%f")

