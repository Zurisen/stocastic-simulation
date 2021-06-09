#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:48:29 2021

@author: carlos
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from time import time
from numpy.random import *
from day3_fun import * 

#%%

## Test distro
n_service_units = 10
mean_service_time = 8
mean_time_between_customers = 1
n_customers = 10000
arrival_dist_type = "poisson"
service_dist_type = "pareto"

# Commented code to check for individual runs
blocked_customers,_,_ = queue_simulation_merged(n_service_units, mean_service_time, mean_time_between_customers, \
											   n_customers, arrival_dist_type, service_dist_type)
print(blocked_customers)


#%%
## Create distribution

n_service_units = 10
mean_service_time = 8
mean_time_between_customers = 1
n_customers = 10000
arrival_dist_type = "poisson"
service_dist_type = "exp"


n_iters = 100
blocked_customers_array = np.zeros(n_iters)
for j in range(n_iters): 
	blocked_customers,_,_ = queue_simulation_merged(n_service_units, mean_service_time, \
											 mean_time_between_customers, n_customers, arrival_dist_type, service_dist_type)
	blocked_customers_array[j] = blocked_customers
blocked_customers_array=blocked_customers_array/(n_customers) # Percentage of rejected customers (per simulation)

mean = np.mean(blocked_customers_array)
std = np.std(blocked_customers_array)

ci = stats.norm.interval(0.95, loc=mean, scale=std) # 95% confidence intervals
plt.hist(blocked_customers_array,50)
plt.title(r"Blocked customers dist")
plt.axvline(x=ci[0], color="black", alpha=0.8,linestyle="--")
plt.axvline(x=ci[1], color="black", alpha=0.8,linestyle="--")

print(mean, std)