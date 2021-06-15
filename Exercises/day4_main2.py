#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:54:47 2021

@author: carlos
"""
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from scipy import stats
from day3_fun import *
from day4_fun import *

#%%
""" Reduce variance on day 3 exercise using control variate """

n_service_units = 10
mean_service_time = 8
mean_time_between_customers = 1
n_customers = 10000
arrival_dist_type = "poisson"
service_dist_type = "exp"


n_iters = 100
blocked_customers_array = np.zeros(n_iters)
mean_arrival_array = np.zeros(n_iters)
for j in range(n_iters): 
	blocked_customers, mean_arrival, _ = queue_simulation_merged(n_service_units, mean_service_time, mean_time_between_customers,\
											  n_customers, arrival_dist_type, service_dist_type)
	blocked_customers_array[j] = blocked_customers
	mean_arrival_array[j] = mean_arrival
	
blocked_customers_array=blocked_customers_array/(n_customers) # Percentage of rejected customers (per simulation)

## Apply mean arrival times as control variate 
Z_mean, Z_std, Z_array = cvar(X=blocked_customers_array, Y=mean_arrival_array)

ci = stats.norm.interval(0.95, loc=Z_mean, scale=Z_std) # 95% confidence intervals
plt.hist(blocked_customers_array,50)
plt.title(r"Blocked customers dist")
plt.axvline(x=ci[0], color="black", alpha=0.8,linestyle="--")
plt.axvline(x=ci[1], color="black", alpha=0.8,linestyle="--")

#%%
""" Importance sampling estimator """

a = 0 # Beginning of the interval
b = 1 # End of the interval
lamb_array = np.linspace(1,10,10000)
exact_value = np.exp(1)-1

error = 300
value_mean = 0
value_std = 0
value_array = 0
for i in range(10000):
	mean, std, arr = exp_importance(a,b, lamb_array[i])
	new_error = np.abs(mean-exact_value)
	if new_error<error:
		error = new_error
		value_mean = mean
		value_std = std
		value_array = arr
		lamb = lamb_array[i]
		
print("  \t\t\tMean\t\t Var\t\t lambda\n", "=======================================")
print( " Computed\t%1.4f\t" %value_mean, "%1.4f\t" %np.power(value_std,2), "%1.4f\n" %lamb)
ci = stats.norm.interval(0.95, loc=value_mean, scale=value_std) # 95% confidence intervals
plt.hist(value_array,50)
plt.title(r"Importance sampling $\int_0^1 e^x$ estimation")
plt.axvline(x=ci[0], color="black", alpha=0.8,linestyle="--")
plt.axvline(x=ci[1], color="black", alpha=0.8,linestyle="--")

#%% 
""" Estimate probability X>a """

## Using crude MC for X>a
a = 1
value_mean, value_std, value_array = normal_crudeMC(a)
print("  \t\t\tMean\t\t Var\n", "==============================")
print( " Computed\t%1.8f\t" %value_mean, "%1.12f\n" %np.power(value_std,2))

# Plot sampled values
N = normal(0,1,size=3000)
in_values = N[N<=a]
out_values = N[N>a]
in_y_random = random(len(in_values))
out_y_random = random(len(out_values))

plt.figure(0)
plt.axvline(x=a, color="black", alpha=1,linestyle="-")
plt.scatter(in_values, in_y_random, color="red")
plt.scatter(out_values, out_y_random, color="blue")
plt.title(r" Scatter plot of 3000 samples from N(0,1)")


# Plot distribution
ci = stats.norm.interval(0.95, loc=value_mean, scale=value_std) # 95% confidence intervals

plt.figure(1)
plt.hist(value_array,50, color="blue")
plt.title(r"Crude MC X>"+str(a)+" estimation")
plt.axvline(x=ci[0], color="black", alpha=0.8,linestyle="--")
plt.axvline(x=ci[1], color="black", alpha=0.8,linestyle="--")
#%%
a = 1
## Using Importance sampling

value_mean, value_std, value_array = normal_importance(a, sigma=1)
print("  \t\t\tMean\t\t Var\n", "==============================")
print( " Computed\t%1.8f\t" %value_mean, "%1.12f\n" %np.power(value_std,2))

# Plot distribution
ci = stats.norm.interval(0.95, loc=value_mean, scale=value_std) # 95% confidence intervals

plt.hist(value_array,50, color="blue")
plt.title(r"Importance sampling X>"+str(a)+" estimation")
plt.axvline(x=ci[0], color="black", alpha=0.8,linestyle="--")
plt.axvline(x=ci[1], color="black", alpha=0.8,linestyle="--")

