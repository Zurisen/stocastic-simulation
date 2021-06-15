#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:01:17 2021

@author: carlos
"""
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from scipy import stats
from day5_fun import *
#%%
""" Truncated Poisson - Metropolis hastings"""
A = 8 ## Arrival intensity * mean service time
m = 10 ## Number of services
X = metropolis_hastings(A,m)

plt.figure()
plt.hist(X,10)
plt.title("Truncated Poisson - Metropolis hastings")
#%%
""" Truncated Poisson - Mixed Metropolis hastings"""
A1 = 4 
A2 = 4
m = 10 ## Number of services
I,J = mixed_metropolis_hastings(A1,A2,m)

plt.figure(0)
plt.hist(I,10, alpha=0.7, label=r"$i$")
plt.hist(J,10, alpha=0.7, label=r"$j$")
plt.title("Truncated Poisson - Mixed Metropolis hastings")
plt.legend()

""" Truncated Poisson - Mixed Metropolis hastings coordinate wise"""
A1 = 4
A2 = 4
m = 10 ## Number of services
I,J = mixed_metropolis_hastings(A1,A2,m, coordinate_wise=True)

plt.figure(1)
plt.hist(I,10, alpha=0.7, label=r"$i$ - Corodinate wise")
plt.hist(J,10, alpha=0.7, label=r"$j$ - Coordinate wise")
plt.title("Truncated Poisson - Mixed Metropolis hastings coordinate wise")

plt.legend()
#%%
""" Truncated Poisson - Gibbs sampling"""
A1 = 4
A2 = 4
m = 10 ## Number of services
I,J = gibbs_sampling(A1,A2,m)

plt.figure(2)
plt.hist(I,10, alpha=0.7, label=r"$i$ - Gibbs sampling")
plt.hist(J,10, alpha=0.7, label=r"$j$ - Gibbs sampling")

plt.title("Truncated Poisson - Gibbs sampling")

plt.legend()

#%% 
""" Bayesian statistical problem """
