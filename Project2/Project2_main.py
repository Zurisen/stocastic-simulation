#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 09:25:30 2021

@author: carlos
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from time import time
from numpy.random import *
from Project2_fun import * 

#%% Without F
realocation_probs = np.array([[0,0.05,0.1,0.05,0.8], [0.2,0,0.5,0.15,0.15], [0.3,0.2,0,0.2,0.3], 
                              [0.35,0.3,0.05,0,0.3], [0.2,0.1,0.6,0.1,0]])

parameters = np.array([[55,14.5,2.9,7],[40,11.0,4.0,5],[30,8.0,4.5,2],[20,6.5,1.4,10],[20,5.0,3.9,5]])

total_iters = 100
n_patients = 5000
ward_distribution = "exp" ## "exp" or "lognormal"

accepted_array = np.zeros((total_iters, parameters.shape[0]))
relocated_array = np.zeros((total_iters, parameters.shape[0]))
rejected_array = np.zeros((total_iters, parameters.shape[0]))
patient_types_count_array = np.zeros((total_iters, parameters.shape[0]))
penalty_array = np.zeros(total_iters)

for n_iter in range(total_iters):
    accepted, relocated, rejected, patient_types_count, penalty = beds_simulation(realocation_probs, parameters,
                                                                                  ward_distribution = ward_distribution, n_patients=n_patients)

    accepted_array[n_iter] = accepted
    relocated_array[n_iter] = relocated
    rejected_array[n_iter] = rejected
    patient_types_count_array[n_iter] = patient_types_count
    penalty_array[n_iter] = penalty
#%%
accepted_percentage_without = accepted_array/patient_types_count_array
relocated_percentage_without = relocated_array/patient_types_count_array
rejected_percentage_without = rejected_array/patient_types_count_array

mean_accepted_without = np.mean(accepted_percentage_without,axis=0) ## percentage of accepted of each type
mean_relocated_without = np.mean(relocated_percentage_without,axis=0) ## percentage of relocated of each type
mean_rejected_without = np.mean(rejected_percentage_without,axis=0) ## percentage of rejected of each type

mean_penalty_without = np.mean(penalty_array)

#%% With F

realocation_probs = np.array([[0,0.05,0.1,0.05,0.8, 0], [0.2,0,0.5,0.15,0.15,0], [0.3,0.2,0,0.2,0.3,0],
                              [0.35,0.3,0.05,0,0.3,0], [0.2,0.1,0.6,0.1,0,0],[0.2,0.2,0.2,0.2,0.2,0]])

#parameters = np.array([[50,14.5,2.9,7],[40,11.0,4.0,5],[30,8.0,4.5,2],[20,6.5,1.4,10],[20,5.0,3.9,5], [5, 13.0, 2.2, 0]])
m = 0
parameters = np.array([[50+m,14.5,2.9,7],[30+m,11.0,4.0,5],[25+m,8.0,4.5,2],[20+m,6.5,1.4,10],[15+m,5.0,3.9,5], [25+m, 13.0, 2.2, 0]])


total_iters = 100
n_patients = 5000
ward_distribution = "exp" ## "exp" or "lognormal"

accepted_array = np.zeros((total_iters, parameters.shape[0]))
relocated_array = np.zeros((total_iters, parameters.shape[0]))
rejected_array = np.zeros((total_iters, parameters.shape[0]))
patient_types_count_array = np.zeros((total_iters, parameters.shape[0]))
penalty_array = np.zeros(total_iters)

for n_iter in range(total_iters):
    accepted, relocated, rejected, patient_types_count, penalty = beds_simulation(realocation_probs, parameters,
                                                                                  ward_distribution = ward_distribution, n_patients=n_patients)

    accepted_array[n_iter] = accepted
    relocated_array[n_iter] = relocated
    rejected_array[n_iter] = rejected
    patient_types_count_array[n_iter] = patient_types_count
    penalty_array[n_iter] = penalty
    
#%%
accepted_percentage_with= accepted_array/patient_types_count_array
relocated_percentage_with= relocated_array/patient_types_count_array
rejected_percentage_with= rejected_array/patient_types_count_array

mean_accepted_with= np.mean(accepted_percentage_with,axis=0) ## percentage of accepted of each type
mean_relocated_with= np.mean(relocated_percentage_with,axis=0) ## percentage of relocated of each type
mean_rejected_with= np.mean(rejected_percentage_with,axis=0) ## percentage of rejected of each type

mean_penalty_with= np.mean(penalty_array)

print(mean_accepted_with)
print(mean_penalty_with)
#%%
# option 1, specify props dictionaries
plt.figure()
positions = np.arange(0,5)
c = "red"
withoutF=plt.boxplot(accepted_percentage_without*100, positions = positions ,notch=True, patch_artist=True,
            boxprops=dict(facecolor=c, color="white"),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color="white"),
            widths=(0.35, 0.35, 0.35, 0.35, 0.35)
            )

c = "purple"
withF=plt.boxplot(accepted_percentage_with[:,:-1]*100, positions=positions+0.35 ,notch=True, patch_artist=True,
            boxprops=dict(facecolor=c, color="white"),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color="white"),
            widths=(0.35, 0.35, 0.35, 0.35, 0.35)
            )

ax = plt.gca()
plt.xticks(positions)
ax.set_xticklabels(["A", "B", "C", "D", "E"])
plt.title("Accepted patients in their correspondant ward")
plt.ylabel("Percentage (%)")
plt.legend([withoutF["boxes"][0], withF["boxes"][0]],["Without F ward", "With F ward"])

plt.savefig("withandwithoutF_comparison.svg", format="svg")

