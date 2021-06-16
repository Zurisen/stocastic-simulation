#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 13:38:48 2021

@author: carlos
"""
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from scipy import stats
from random import sample
from Exs1_fun import *
from Exs2_fun import *
from scipy.linalg import expm
#%%
######################
##### EXERCISE 7 #####
######################

Q = np.array([[-0.0085, 0.005, 0.0025, 0, 0.001],
              [0,-0.014,0.005,0.004,0.005],
              [0,0,-0.008,0.003,0.005],
              [0,0,0,-0.009,0.009],
              [0,0,0,0,0]])
n_women = 1000
#%%
""" Simulate 1000 women Exercise 7"""

last_states, women_months = simulate_death_2(Q, n_women, limit_months=30.5)

mean_month = np.mean(women_months)
std_month = np.std(women_months)
ci = stats.norm.interval(0.95, loc=mean_month, scale=std_month) # 95% confidence intervals
plt.hist(women_months,bins=20)
plt.title("1000 women simulation Exercise 7")
plt.axvline(x=ci[0], color="black", alpha=0.8,linestyle="--")
plt.axvline(x=ci[1], color="black", alpha=0.8,linestyle="--")
plt.xlabel("Month of death")
plt.ylabel("Number of women")

print("Mean month of death: %1.2f" %mean_month)
print("Std month of death: %1.4f" %std_month)

## Count proportion of women standing in states 3 or 4 (distant cancer reappear) int the month 30.5
distant_cancer = np.sum(np.isin(last_states,[2,3])) 
print("Number of women with distant cancer at month 30.5: %d" %distant_cancer,"out of %d" %n_women ,"(%1.2f%%)" %(distant_cancer/n_women*100))

#%% 
""" Analytical distribution Exercise 7 """
CDF_emp, _, _= plt.hist(women_months, 100, density=True, histtype='step',
                           cumulative=True, label='Empirical')
t_an = np.linspace(0,max(women_months), 100)
CDF_an = np.zeros(len(t_an))
## Now generate CDF of lifetime (month of death) from the analytical result 
Qs = Q[:-1,:-1]

for i in range(len(t_an)):
    CDF_an[i] = 1- np.matmul( np.array([0,1,0,0]).T,np.matmul( expm(Qs*t_an[i]), np.ones(Qs.shape[1]) ) )
    
plt.plot(t_an,CDF_an, label="CDF analytical")
plt.xlim((0,t_an[-1]))

ks = stats.kstest(CDF_emp,CDF_an)