#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 09:57:11 2021

@author: carlos
"""
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from scipy import stats
from day6_fun import *
#%%
M_cost = mat2()

total_cost, route, T = min_cost_overall(M_cost,1,5)


fig, ax = plt.subplots(2,1, figsize=(6,10))
ax[0].plot(total_cost,T)
ax[0].set_ylabel("Temperature")
ax[1].hist(total_cost)
ax[1].set_ylabel("Counts")
ax[1].set_xlabel("Total cost")

#%%
a = -5
b = -a
X = np.array([56,101,78,67,93,87,64,72,80,69])

P, boot_value, Xmean, Xvar, _, _ = ex13(a,b,X)

plt.hist(boot_value)

#%%
a = -5
b = -a
X = np.array([5,4,9,6,21,17,11,20,7,10,21,15,13,16,8])

P, boot_value, Xmean, Xvar, _, _ = ex13(a,b,X)

plt.hist(boot_value)

#%%
boot_means, boot_vars, boot_medians = pareto_boot()
print(" Bootstrap over Pareto distribution\n", "==================================")
print(" Mean: %1.4f\t" %np.mean(boot_means),"Var: %1.4f\t" %np.var(boot_means),"Median: %1.4f\n" %np.median(boot_means))
plt.hist(boot_means,50)
plt.title("Pareto bootstrap means")

## b)
plt.figure()
meanboot_means, meanboot_vars, meanboot_medians = vector_boot(boot_means)
plt.hist(meanboot_means,50)
plt.title("Boostrap Means over Pareto means")
print(" Bootstrap over Pareto means\n", "==================================")
print(" Mean: %1.4f\t" %np.mean(meanboot_means),"Var: %1.4f\t" %np.var(meanboot_means),"Median: %1.4f\n" %np.median(meanboot_means))

## b)
plt.figure()
medianboot_means, medianboot_vars, medianboot_medians = vector_boot(boot_medians)
plt.hist(medianboot_means,50)
plt.title("Boostrap Means over Pareto medians")
print(" Bootstrap over Pareto medians\n", "==================================")
print(" Mean: %1.4f\t" %np.mean(medianboot_means),"Var: %1.4f\t" %np.var(medianboot_means),"Median: %1.4f\n" %np.median(medianboot_means))