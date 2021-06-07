#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:28:23 2021

@author: carlos
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
#%% 
""" Linear Congruential generator (LCG)"""

M = 16
a = 5
c = 1
x0 = 3

rnd_len = 10

def LCG(x0,n,M,a,c):
	rnd_gen = np.array([x0])
	rnd_gen[0] = x0
	for i in range(rnd_len):
		rnd_gen = np.append(rnd_gen,(c+a*rnd_gen[-1])%M )
	return rnd_gen

print(LCG(x0,rnd_len,M,a,c))

#%%
""" 10000 samples and histogram """

M = 16
a = 5
c = 1
x0 = 3

rnd_len = 10000
sol = LCG(x0,rnd_len,M,a,c)
plt.hist(sol, 16) ## 16 is the maximum integer, so the distribution should be even at 16 bins

#%% 
""" Evaluate statistics """

rnd_len = 10000
a = 5
c = 1
x0 = 3

rnd_len = 10000
n_classes = 16 # == M

def chi2(n_classes, n_observed, n_expected):
	T = 0
	for i in range(n_classes):
		T = (n_observed[i]-n_expected[i])/n_expected[i]
	return T

numbers = LCG(x0,rnd_len,n_classes,a,c)

print("Chi^2: ", chi2(n_classes,numbers,[rnd_len/n_classes]*16))
print("p-value: ", stats.chisquare(numbers))

#%%
""" Kolmogorov-Smirnov """

def KS(x,numbers):
	KS = sum(numbers<=x)/len(numbers)
	return KS

numbers = LCG(x0,rnd_len,n_classes,a,c)
x = 5

ksplot = np.zeros(n_classes)
for i in range(n_classes):
	ksplot[i] = (KS(i,numbers))

plt.plot(ksplot)	
	


