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

total_iters = 10
n_patients = 16500

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

# Generate matrix to latex
results_without = np.zeros((len(mean_accepted_without), 5))
results_without[:,0] = np.array(["1", "2", "3", "4", "5"])
results_without[:,1] = mean_accepted_without
results_without[:,2] = mean_relocated_without
results_without[:,3] = mean_rejected_without 
results_without[:,4] = mean_penalty_without

#np.savetxt('withoutFward_results.csv', results_without, delimiter=',')

print(mean_penalty_without)

#%% With F

realocation_probs = np.array([[0,0.05,0.1,0.05,0.8, 0], [0.2,0,0.5,0.15,0.15,0], [0.3,0.2,0,0.2,0.3,0],
                              [0.35,0.3,0.05,0,0.3,0], [0.2,0.1,0.6,0.1,0,0],[0.2,0.2,0.2,0.2,0.2,0]])

#parameters = np.array([[50,14.5,2.9,7],[40,11.0,4.0,5],[30,8.0,4.5,2],[20,6.5,1.4,10],[20,5.0,3.9,5], [5, 13.0, 2.2, 0]])
parameters = np.array([[50,14.5,2.9,7],[25,11.0,4.0,5],[20,8.0,4.5,2],[20,6.5,1.4,10],[15,5.0,3.9,5], [35, 13.0, 2.2, 0]])


total_iters = 10
n_patients = 16500
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

# Generate matrix to latex
results_with= np.zeros((len(mean_accepted_with), 5))
results_with[:,0] = np.array(["1", "2", "3", "4", "5", "6"])
results_with[:,1] = mean_accepted_with
results_with[:,2] = mean_relocated_with
results_with[:,3] = mean_rejected_with
results_with[:,4] = mean_penalty_with

np.savetxt('withFward_results.csv', results_with, delimiter=',')

print(mean_penalty_with)

#%%
plt.figure()
ax1 = plt.gca()

xaxis = np.array([0,2,4,6,8])
ax1.bar(xaxis,mean_rejected_without,alpha=1, color="#673147",label="Rejected",width=1)
ax1.bar(xaxis,mean_relocated_without, bottom=mean_rejected_without ,alpha=0.8, color="#800080",label="Relocated",width=1)
ax1.bar(xaxis,mean_accepted_without, bottom=mean_relocated_without+mean_rejected_without ,alpha=1, color="#DA70D6",label="Accepted",width=1)
plt.xticks(xaxis)
plt.title("Patients attendance per ward (wihout F ward)")
plt.legend(bbox_to_anchor=(1.01, 1.02), loc='upper left')
ax1.set_xticklabels(["A", "B", "C", "D", "E"])
plt.savefig("withoutFward_distr.svg", format="svg")


#%%
plt.figure()
ax2 = plt.gca()

xaxis = np.array([0,2,4,6,8,10])
ax2.bar(xaxis,mean_rejected_with,alpha=1, color="#355E3B",label="Rejected",width=1)
ax2.bar(xaxis,mean_relocated_with, bottom=mean_rejected_with ,alpha=0.8, color="#008000",label="Relocated",width=1)
ax2.bar(xaxis,mean_accepted_with, bottom=mean_relocated_with+mean_rejected_with ,alpha=0.8, color="#00FF7F",label="Accepted",width=1)
plt.xticks(xaxis)
plt.title("Patients attendance per ward (with F ward)")
plt.legend(bbox_to_anchor=(1.01, 1.02), loc='upper left')

ax2.set_xticklabels(["A", "B", "C", "D", "E", "$F^*$"])
plt.savefig("withandwithoutF_comparison.svg", format="svg")
plt.savefig("withFward_distr.svg", format="svg")

#%%

####################
####################
# Log normal distribution on ward attendance

realocation_probs = np.array([[0,0.05,0.1,0.05,0.8, 0], [0.2,0,0.5,0.15,0.15,0], [0.3,0.2,0,0.2,0.3,0],
                              [0.35,0.3,0.05,0,0.3,0], [0.2,0.1,0.6,0.1,0,0],[0.2,0.2,0.2,0.2,0.2,0]])

#parameters = np.array([[50,14.5,2.9,7],[40,11.0,4.0,5],[30,8.0,4.5,2],[20,6.5,1.4,10],[20,5.0,3.9,5], [5, 13.0, 2.2, 0]])
parameters = np.array([[50,14.5,2.9,7],[25,11.0,4.0,5],[20,8.0,4.5,2],[20,6.5,1.4,10],[15,5.0,3.9,5], [35, 13.0, 2.2, 0]])


trial_sigmas = np.linspace(1,10,20)
total_iters = len(trial_sigmas)
n_patients = 16500
ward_distribution = "lognormal" ## "exp" or "lognormal"

accepted_array = np.zeros((total_iters, parameters.shape[0]))
relocated_array = np.zeros((total_iters, parameters.shape[0]))
rejected_array = np.zeros((total_iters, parameters.shape[0]))
patient_types_count_array = np.zeros((total_iters, parameters.shape[0]))
penalty_array = np.zeros(total_iters)

for n_iter in range(total_iters):
    accepted, relocated, rejected, patient_types_count, penalty = beds_simulation(realocation_probs, parameters, sigma = trial_sigmas[n_iter],
                                                                                  ward_distribution = ward_distribution, n_patients=n_patients)

    accepted_array[n_iter] = accepted/patient_types_count
    relocated_array[n_iter] = relocated/patient_types_count
    rejected_array[n_iter] = rejected/patient_types_count
    patient_types_count_array[n_iter] = patient_types_count
    penalty_array[n_iter] = penalty

#%%
plt.figure()
labels=["A", "B", "C", "D", "E", "F"]
for i in range(parameters.shape[0]):
    plt.plot(trial_sigmas, relocated_array[:,i]*100, label="Ward "+ labels[i])
    
ax = plt.gca()
plt.ylabel("Relocated patients (%)")
plt.xlabel("$k$")
plt.legend()

plt.savefig("relocated_vs_k.svg", format="svg")

plt.figure()
for i in range(parameters.shape[0]):
    plt.plot(trial_sigmas, accepted_array[:,i]*100, label="Ward "+ labels[i])
    
ax = plt.gca()
plt.ylabel("Accepted patients (%)")
plt.xlabel("$k$")
plt.legend()

plt.savefig("accepted_vs_k.svg", format="svg")

plt.figure()
for i in range(parameters.shape[0]):
    plt.plot(trial_sigmas, rejected_array[:,i]*100, label="Ward "+ labels[i])
    
ax = plt.gca()
plt.ylabel("Rejected patients (%)")
plt.xlabel("$k$")
plt.legend()

plt.savefig("rejection_vs_k.svg", format="svg")


#%%
####################
####################
# Check how rejections change with adding more beds
realocation_probs = np.array([[0,0.05,0.1,0.05,0.8, 0], [0.2,0,0.5,0.15,0.15,0], [0.3,0.2,0,0.2,0.3,0],
                              [0.35,0.3,0.05,0,0.3,0], [0.2,0.1,0.6,0.1,0,0],[0.2,0.2,0.2,0.2,0.2,0]])

#parameters = np.array([[50,14.5,2.9,7],[40,11.0,4.0,5],[30,8.0,4.5,2],[20,6.5,1.4,10],[20,5.0,3.9,5], [5, 13.0, 2.2, 0]])

parameters = np.array([[50,14.5,2.9,7],[25,11.0,4.0,5],[20,8.0,4.5,2],[20,6.5,1.4,10],[15,5.0,3.9,5], [35, 13.0, 2.2, 0]])
new_parameters = np.array([[50,14.5,2.9,7],[25,11.0,4.0,5],[20,8.0,4.5,2],[20,6.5,1.4,10],[15,5.0,3.9,5], [35, 13.0, 2.2, 0]])

n_patients = 16500
ward_distribution = "exp" ## "exp" or "lognormal"
add_beds = np.arange(-5,10,1)
total_iters = len(add_beds)
total_beds = np.zeros(total_iters)

accepted_array = np.zeros((total_iters, parameters.shape[0]))
relocated_array = np.zeros((total_iters, parameters.shape[0]))
rejected_array = np.zeros((total_iters, parameters.shape[0]))
patient_types_count_array = np.zeros((total_iters, parameters.shape[0]))
penalty_array = np.zeros(total_iters)


for n_iter in range(total_iters):
    new_parameters[:,0] = parameters[:,0] + add_beds[n_iter]
    total_beds[n_iter] = np.sum(new_parameters[:,0])
    accepted, relocated, rejected, patient_types_count, penalty = beds_simulation(realocation_probs, new_parameters,
                                                                                  ward_distribution = ward_distribution, n_patients=n_patients)

    accepted_array[n_iter] = accepted/patient_types_count
    relocated_array[n_iter] = relocated/patient_types_count
    rejected_array[n_iter] = rejected/patient_types_count
    patient_types_count_array[n_iter] = patient_types_count
    penalty_array[n_iter] = penalty
    
#%%
plt.figure()
labels=["A", "B", "C", "D", "E", "F"]
for i in range(parameters.shape[0]):
    plt.plot(total_beds, relocated_array[:,i]*100, label="Ward "+ labels[i])
    
ax = plt.gca()
plt.ylabel("Relocated patients (%)")
plt.xlabel("Total number of beds")
plt.legend()

plt.savefig("relocated_vs_nbeds.svg", format="svg")

plt.figure()
for i in range(parameters.shape[0]):
    plt.plot(total_beds, accepted_array[:,i]*100, label="Ward "+ labels[i])
    
ax = plt.gca()
plt.ylabel("Accepted patients (%)")
plt.xlabel("Total number of beds")
plt.legend()

plt.savefig("accepted_vs_nbeds.svg", format="svg")

plt.figure()
for i in range(parameters.shape[0]):
    plt.plot(total_beds, rejected_array[:,i]*100, label="Ward "+ labels[i])
    
ax = plt.gca()
plt.ylabel("Rejected patients (%)")
plt.xlabel("Total number of beds")
plt.legend()

plt.savefig("rejection_vs_nbeds.svg", format="svg")


#%%
########################
########################
########################
# Tests of sensitivity


realocation_probs = np.array([[0,0.05,0.1,0.05,0.8, 0], [0.2,0,0.5,0.15,0.15,0], [0.3,0.2,0,0.2,0.3,0],
                              [0.35,0.3,0.05,0,0.3,0], [0.2,0.1,0.6,0.1,0,0],[0.2,0.2,0.2,0.2,0.2,0]])

#parameters = np.array([[50,14.5,2.9,7],[40,11.0,4.0,5],[30,8.0,4.5,2],[20,6.5,1.4,10],[20,5.0,3.9,5], [5, 13.0, 2.2, 0]])
parameters = np.array([[55,14.5,2.9,7],[5,11.0,4.0,5],[10,8.0,4.5,2],[60,6.5,1.4,10],[5,5.0,3.9,5], [35, 13.0, 2.2, 0]])


total_iters = 10
n_patients = 16500
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
    
accepted_percentage_with= accepted_array/patient_types_count_array
relocated_percentage_with= relocated_array/patient_types_count_array
rejected_percentage_with= rejected_array/patient_types_count_array

mean_accepted_with= np.mean(accepted_percentage_with,axis=0) ## percentage of accepted of each type
mean_relocated_with= np.mean(relocated_percentage_with,axis=0) ## percentage of relocated of each type
mean_rejected_with= np.mean(rejected_percentage_with,axis=0) ## percentage of rejected of each type

mean_penalty_with= np.mean(penalty_array)

# Generate matrix to latex
results_with= np.zeros((len(mean_accepted_with), 5))
results_with[:,0] = np.array(["1", "2", "3", "4", "5", "6"])
results_with[:,1] = mean_accepted_with
results_with[:,2] = mean_relocated_with
results_with[:,3] = mean_rejected_with
results_with[:,4] = mean_penalty_with

np.savetxt('test1_results.csv', results_with, delimiter=',')

print(mean_penalty_with)

plt.figure()
ax2 = plt.gca()

xaxis = np.array([0,2,4,6,8,10])
ax2.bar(xaxis,mean_rejected_with,alpha=1, color="#355E3B",label="Rejected",width=1)
ax2.bar(xaxis,mean_relocated_with, bottom=mean_rejected_with ,alpha=0.8, color="#008000",label="Relocated",width=1)
ax2.bar(xaxis,mean_accepted_with, bottom=mean_relocated_with+mean_rejected_with ,alpha=0.8, color="#00FF7F",label="Accepted",width=1)
plt.xticks(xaxis)
plt.title("Patients attendance per ward (with F ward)")
plt.legend(bbox_to_anchor=(1.01, 1.02), loc='upper left')

ax2.set_xticklabels(["A", "B", "C", "D", "E", "$F^*$"])
plt.savefig("test1_results.svg", format="svg")

#%%
########################
########################
########################
# Tests of sensitivity


realocation_probs = np.array([[0,0.05,0.1,0.05,0.8, 0], [0.2,0,0.5,0.15,0.15,0], [0.3,0.2,0,0.2,0.3,0],
                              [0.35,0.3,0.05,0,0.3,0], [0.2,0.1,0.6,0.1,0,0],[0.2,0.2,0.2,0.2,0.2,0]])

#parameters = np.array([[50,14.5,2.9,7],[40,11.0,4.0,5],[30,8.0,4.5,2],[20,6.5,1.4,10],[20,5.0,3.9,5], [5, 13.0, 2.2, 0]])
parameters = np.array([[5,14.5,2.9,7],[40,11.0,4.0,5],[40,8.0,4.5,2],[5,6.5,1.4,10],[40,5.0,3.9,5], [35, 13.0, 2.2, 0]])


total_iters = 10
n_patients = 16500
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
    
accepted_percentage_with= accepted_array/patient_types_count_array
relocated_percentage_with= relocated_array/patient_types_count_array
rejected_percentage_with= rejected_array/patient_types_count_array

mean_accepted_with= np.mean(accepted_percentage_with,axis=0) ## percentage of accepted of each type
mean_relocated_with= np.mean(relocated_percentage_with,axis=0) ## percentage of relocated of each type
mean_rejected_with= np.mean(rejected_percentage_with,axis=0) ## percentage of rejected of each type

mean_penalty_with= np.mean(penalty_array)

# Generate matrix to latex
results_with= np.zeros((len(mean_accepted_with), 5))
results_with[:,0] = np.array(["1", "2", "3", "4", "5", "6"])
results_with[:,1] = mean_accepted_with
results_with[:,2] = mean_relocated_with
results_with[:,3] = mean_rejected_with
results_with[:,4] = mean_penalty_with

np.savetxt('test2_results.csv', results_with, delimiter=',')

print(mean_penalty_with)

plt.figure()
ax2 = plt.gca()

xaxis = np.array([0,2,4,6,8,10])
ax2.bar(xaxis,mean_rejected_with,alpha=1, color="#355E3B",label="Rejected",width=1)
ax2.bar(xaxis,mean_relocated_with, bottom=mean_rejected_with ,alpha=0.8, color="#008000",label="Relocated",width=1)
ax2.bar(xaxis,mean_accepted_with, bottom=mean_relocated_with+mean_rejected_with ,alpha=0.8, color="#00FF7F",label="Accepted",width=1)
plt.xticks(xaxis)
plt.title("Patients attendance per ward (with F ward)")
plt.legend(bbox_to_anchor=(1.01, 1.02), loc='upper left')

ax2.set_xticklabels(["A", "B", "C", "D", "E", "$F^*$"])
plt.savefig("test2_results.svg", format="svg")











