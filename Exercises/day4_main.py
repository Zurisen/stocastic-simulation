#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:48:29 2021

@author: carlos
"""
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from scipy import stats
from day4_fun import * 
#%%
""" Crude Monte-Carlo estimator """

a = 0 # Beginning of the interval
b = 1 # End of the interval

" First we try over one only trapezoidal partition and compare to the analytical results from the slides"
value_mean, value_std, value_array = exp_crudeMC(a,b,n_uni=1)
exact_mean = np.exp(1)-1
var_mean = 0.5*(np.exp(2)-1)-np.power(np.exp(1)-1,2)

relerror_mean = np.abs((value_mean-exact_mean)/exact_mean)*100
relerror_var=np.abs( (np.power(value_std,2) - var_mean)/var_mean)*100

print(" For n_uni = 1, n_iter = 10000\n", "==============================")
print("  \t\t\tMean\t\t Var\n", "==============================")
print( " Computed\t%1.4f\t" %value_mean, "%1.4f\n" %np.power(value_std,2))
print( " Exact\t\t%1.4f\t" %exact_mean, "%1.4f\n" %var_mean)
print( " Relerror(%%)\t%1.4f\t" %relerror_mean, "%1.4f\n" %relerror_var)


" Now with n_uni=100 as stated in the class exercise"
value_mean, value_std, value_array = exp_crudeMC(a,b)
print(" \n For n_uni = 100, n_iter = 10000\n", "==============================")
print("  \t\t\tMean\t\t Var\n", "==============================")
print( " Computed\t%1.4f\t" %value_mean, "%1.4f\n" %np.power(value_std,2))
ci = stats.norm.interval(0.95, loc=value_mean, scale=value_std) # 95% confidence intervals
plt.hist(value_array,50)
plt.title(r"Crude MC $\int_0^1 e^x$ estimation")
plt.axvline(x=ci[0], color="black", alpha=0.8,linestyle="--")
plt.axvline(x=ci[1], color="black", alpha=0.8,linestyle="--")

#%%
""" Antithetic variables estimator """

a = 0 # Beginning of the interval
b = 1 # End of the interval

" First we try over one only n_uni=1 partition and compare to the analytical results from the slides"

value_mean, value_std, value_array = exp_antithetic(a,b,n_uni=1)
exact_mean = np.exp(1)-1
var_mean = 0.5*( 0.5*(np.exp(2)-1)-np.power(np.exp(1)-1,2) ) +0.5*( 3*np.exp(1) -np.exp(2) - 1 )

relerror_mean = np.abs((value_mean-exact_mean)/exact_mean)*100
relerror_var=np.abs( (np.power(value_std,2) - var_mean)/var_mean)*100

print(" For n_uni= 1, n_iter = 10000\n", "==============================")
print("  \t\t\tMean\t\t Var\n", "==============================")
print( " Computed\t%1.4f\t" %value_mean, "%1.4f\n" %np.power(value_std,2))
print( " Exact\t\t%1.4f\t" %exact_mean, "%1.4f\n" %var_mean)
print( " Relerror(%%)\t%1.4f\t" %relerror_mean, "%1.4f\n" %relerror_var)

" Now with n_uni=100 as stated in the class exercise"

value_mean, value_std, value_array = exp_antithetic(a,b)
print(" \n For n_uni= 100, n_iter = 10000\n", "==============================")
print("  \t\t\tMean\t\t Var\n", "==============================")
print( " Computed\t%1.4f\t" %value_mean, "%1.8f\n" %np.power(value_std,2))
ci = stats.norm.interval(0.95, loc=value_mean, scale=value_std) # 95% confidence intervals
plt.hist(value_array,50)
plt.title(r"Antithetic variables $\int_0^1 e^x$ estimation")
plt.axvline(x=ci[0], color="black", alpha=0.8,linestyle="--")
plt.axvline(x=ci[1], color="black", alpha=0.8,linestyle="--")

#%%
""" Control variable estimator """

a = 0 # Beginning of the interval
b = 1 # End of the interval

" Compare to the analytical results from the slides"

value_mean, value_std, value_array = exp_cvar(a,b)
exact_mean = np.exp(1)-1
var_mean = 0.5*( 0.5*(np.exp(2)-1)-np.power(np.exp(1)-1,2) ) +0.5*( 3*np.exp(1) -np.exp(2) - 1 )

relerror_mean = np.abs((value_mean-exact_mean)/exact_mean)*100
relerror_var=np.abs( (np.power(value_std,2) - var_mean)/var_mean)*100

print("  \t\t\tMean\t\t Var\n", "==============================")
print( " Computed\t%1.4f\t" %value_mean, "%1.4f\n" %np.power(value_std,2))
print( " Exact\t\t%1.4f\t" %exact_mean, "%1.4f\n" %var_mean)
print( " Relerror(%%)\t%1.4f\t" %relerror_mean, "%1.4f\n" %relerror_var)

" Show histogram and confidence intervals "

ci = stats.norm.interval(0.95, loc=value_mean, scale=value_std) # 95% confidence intervals
plt.hist(value_array,50)
plt.title(r"Control variable $\int_0^1 e^x$ estimation")
plt.axvline(x=ci[0], color="black", alpha=0.8,linestyle="--")
plt.axvline(x=ci[1], color="black", alpha=0.8,linestyle="--")

#%%
""" Stratified sampling estimator """

a = 0 # Beginning of the interval
b = 1 # End of the interval

value_mean, value_std, value_array = exp_stratified(a,b)
print(" \n For n_uni= 100, n_iter = 10\n", "==============================")
print("  \t\t\tMean\t\t Var\n", "==============================")
print( " Computed\t%1.4f\t" %value_mean, "%1.8f\n" %np.power(value_std,2))
ci = stats.norm.interval(0.95, loc=value_mean, scale=value_std) # 95% confidence intervals
plt.hist(value_array,50)
plt.title(r"Stratified sampling $\int_0^1 e^x$ estimation")
plt.axvline(x=ci[0], color="black", alpha=0.8,linestyle="--")
plt.axvline(x=ci[1], color="black", alpha=0.8,linestyle="--")