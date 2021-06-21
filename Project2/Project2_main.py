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

#%% With F

realocation_probs = np.array([[0,0.05,0.1,0.05,0.8, 0], [0.2,0,0.5,0.15,0.15,0], [0.3,0.2,0,0.2,0.3,0],
                              [0.35,0.3,0.05,0,0.3,0], [0.2,0.1,0.6,0.1,0,0],[0.2,0.2,0.2,0.2,0.2,0]])

parameters = np.array([[55,14.5,2.9,7],[40,11.0,4.0,5],[30,8.0,4.5,2],[20,6.5,1.4,10],[20,5.0,3.9,5], [20, 13.0, 2.2, 9]])

#%%

total_iters = 100
n_patients = 5000
urgency_weight = True
ward_distribution = "exp" ## "exp" or "lognormal"

accepted_array = np.zeros((total_iters, parameters.shape[0]))
relocated_array = np.zeros((total_iters, parameters.shape[0]))
rejected_array = np.zeros((total_iters, parameters.shape[0]))
patient_types_count_array = np.zeros((total_iters, parameters.shape[0]))
penalty_array = np.zeros(total_iters)
for n_iter in range(total_iters):
    accepted, relocated, rejected, patient_types_count, penalty = beds_simulation(realocation_probs, parameters, ward_distribution = ward_distribution,
                                                                                  n_patients=n_patients, urgency_weight=urgency_weight)

    accepted_array[n_iter] = accepted
    relocated_array[n_iter] = relocated
    rejected_array[n_iter] = rejected
    patient_types_count_array[n_iter] = patient_types_count
    penalty_array[n_iter] = penalty
#%%
accepted_percentage = accepted_array/patient_types_count_array
relocated_percentage = relocated_array/patient_types_count_array
rejected_percentage = rejected_array/patient_types_count_array

mean_accepted = np.mean(accepted_percentage,axis=0) ## percentage of accepted of each type
mean_relocated = np.mean(relocated_percentage,axis=0) ## percentage of relocated of each type
mean_rejected = np.mean(rejected_percentage,axis=0) ## percentage of rejected of each type

mean_penalty = np.mean(penalty_array)

print(mean_rejected)
print(mean_penalty)