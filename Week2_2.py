#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:25:05 2021

@author: carlos
"""
import numpy as np
from numpy.random import uniform as uni
import matplotlib.pyplot as plt
import scipy.stats as stats

#%%
""" Exponential - inverse transform """
def exp_invtrans(lamb, U_size=100000):
	U = np.random.uniform(size=U_size)
	X = -np.log(U)/lamb
	return X

lamb = 1
X = exp_invtrans(lamb)

plt.hist(X,100)
plt.title(r"Exponential PDF | $\lambda$ = " + str(lamb))

#%%
""" Uniform distribution - inverse transform """
def uni_invtrans(a, b, U_size=100000):
	U = np.random.uniform(size=U_size)
	X = a + (b-a)*U
	return X

a = 0
b = 1
X = uni_invtrans(a, b)

plt.hist(X,100)
plt.title(r"Uniform PDF | a = " + str(a) +", b = " + str(b))

#%%
""" Standard normal distribution - Box-Muller """
def normal_BM(U_size=100000):
	U1 = np.random.uniform(size=U_size)
	U2 = np.random.uniform(size=U_size)
	Z0 = np.sqrt(-2*np.log(U1))*np.cos(2*3.1416*U2)
	Z1 = np.sqrt(-2*np.log(U1))*np.sin(2*3.1416*U2)
	
	return Z0, Z1
	
Z0, Z1 = normal_BM()
plt.hist(Z0,100, alpha=0.5, label=r"$Z_0$")
plt.hist(Z1,100, alpha=0.5, label=r"$Z_1$")
plt.title(r"Standard Normal PDF")
plt.legend()

#%%
""" Standard normal distribution - Box-Muller acception rejection """
def normal_BM_acceptionrejection(U_size=100000):
	U1 = np.random.uniform(0,1,size=U_size)
	V1 = np.random.uniform(-1,1,size=U_size)
	V2 = np.random.uniform(-1,1,size=U_size)
	
	R2 = np.power(V1,2) + np.power(V2,2)
	
	## Accept reject according to condition
	rejected = sum(R2>=1) #count rejected
	accepted = U_size-rejected
	
	V1 = V1[R2<1]
	U1 = U1[R2<1]
	V2 = V2[R2<1]
	R2 = R2[R2<1]
	
	Z0 = np.sqrt(-2*np.log(U1))*V1/np.sqrt(R2)
	Z1 = np.sqrt(-2*np.log(U1))*V2/np.sqrt(R2)
	
	return Z0, Z1, accepted, rejected, U_size

Z0, Z1, accepted, rejected, total = normal_BM()
print("Rejected %d samples" %rejected, "out of %d" %total, "(%1.2f%%)" %(rejected/total*100))
plt.hist(Z0,100, alpha=0.5, label=r"$Z_0$")
plt.hist(Z1,100, alpha=0.5, label=r"$Z_1$")
plt.title(r"Standard Normal PDF")
plt.legend()

#%% 
""" Pareto distribution - inverse transform """
def pareto_invtrans(beta, k, U_size=100000):
	U = np.random.uniform(size=U_size)
	X = beta*(np.power(U, -1/k))
	return X

beta = 1
k = 4
X = pareto_invtrans(beta, k)

plt.hist(X,100)
plt.title(r"Pareto PDF | $\beta$ = " + str(beta) + ", k = " + str(k))

print(" Computed params\n", "=================")
print(" Mean: %1.4f" %np.mean(X))
print(" Var: %1.4f" %np.std(X)**2)
print("\n")

mean_anal = beta*k/(k-1)
std_anal = beta**2*k/((k-1)**2 *(k-2))
print(" Analytical params\n", "=================")
print(" Mean: %1.4f" %mean_anal )
print(" Var: %1.4f" %std_anal )

#%%

Z0, Z1 = normal_BM()
ci = stats.norm.interval(0.95, loc=np.mean(Z0), scale=np.std(Z0))
Z0, Z1 = normal_BM()
plt.hist(Z0,100, alpha=0.5, label=r"$Z_0$")
plt.hist(Z1,100, alpha=0.5, label=r"$Z_1$")
plt.title(r"Standard Normal PDF")
plt.axvline(x=ci[0], color="black", alpha=0.8,linestyle="--")
plt.axvline(x=ci[1], color="black", alpha=0.8,linestyle="--")
plt.legend()
