#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 09:23:51 2021

@author: carlos
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from time import time
from numpy.random import *


def ward_sample(size, scale, distr_type):
    if distr_type == "exp":
        out = exponential(size=size, scale = scale)
    elif distr_type == "lognormal":
        out = lognormal(sigma=4*scale**2, size=size)
    else:
        raise NotImplementedError("Not implemented distribution type: choose exp or lognormal")
    return out
        

def beds_simulation(realoc_probs, params, ward_distribution = "exp", n_patients=5000):
    if realoc_probs.shape[0] != params.shape[0]:
        raise SyntaxError("Dimensions of parameters and realocation probabilites do not match!")        
        
    n_types = params.shape[0]
    bed_capacities = params[:,0]
    arrival_rates = params[:,1]
    ward_rates = params[:,2]
    urgency_pts = params[:,3]    
    
    ## Build arrival times
    arrival_time_dist = np.zeros(n_patients)
    arrival_types = np.zeros(n_patients) ## To store the type of the patient and assign it to a ward
    patient_types_count = np.zeros(n_types) ## store total number of patients in each category
    
    arrival_times_sampler = np.zeros((n_types, n_patients))
    for i in range(n_types):
        arrival_times_sampler[i] = exponential(size=[n_patients], scale=1/arrival_rates[i])
    
    for i in range(n_patients):
        patient_type = np.where( arrival_times_sampler[:,i] == np.min(arrival_times_sampler[:,i]) )[0]
        patient_types_count[patient_type] +=1
        
        arrival_time_dist[i] = np.min(arrival_times_sampler[:,i])
        arrival_types[i] = patient_type
    arrival_times = np.cumsum(arrival_time_dist)

    ## Build ward service times
    ward_time_dist = np.zeros((n_types, n_patients))
    for i in range(n_types):
        #ward_time_dist[i] = exponential(size=[n_patients], scale = ward_rates[i])
        ward_time_dist[i] = ward_sample(distr_type=ward_distribution, size=n_patients, scale=ward_rates[i])
    
    ## Start counter of beds occupation for each ward
    beds_occupied = np.zeros(n_types)
    
    ## Start counter for rejected patients
    rejected = np.zeros(n_types)
    ## Start counter for relocated patients
    relocated = np.zeros(n_types)
    ## Start counter for patients alocated in their correct ward
    accepted = np.zeros(n_types)
    
    ## Count total penalty 
    penalty = 0
    
    patients_in_bed_times = np.array([0])
    patients_in_bed_type = np.array([0])
    
    for n in range(n_patients):   
        
        n_patient_type = int(arrival_types[n]) ## keep track of the patient type in this iteration
        # just_freed_beds = np.where(ward_times<arrival_times[n])[0] ## Check beds that has just been freed   
        
        cured_patients = np.where(patients_in_bed_times < arrival_times[n])[0]
        
        patients_in_bed_times = np.delete(patients_in_bed_times, cured_patients)                   
        patients_in_bed_type = np.delete(patients_in_bed_type, cured_patients) 
        
        for i in range(n_types):
            beds_occupied[i] = np.sum(patients_in_bed_type == i) 
        
        if (beds_occupied[n_patient_type] < bed_capacities[n_patient_type]): ## Check if the correct ward for the patient has available beds
            accepted[n_patient_type] += 1
            
            ward_time = ward_time_dist[n_patient_type,n] + arrival_times[n]
            patients_in_bed_times = np.append(patients_in_bed_times, ward_time)
            patients_in_bed_type = np.append(patients_in_bed_type, n_patient_type)
            
        elif np.sum(beds_occupied<bed_capacities)>0: ## Check if there is any other ward with beds available
            relocated[n_patient_type] += 1
            penalty += urgency_pts[n_patient_type]
            wards_available = np.where(beds_occupied<bed_capacities)[0]
            probs = realoc_probs[n_patient_type,wards_available]
            
            if np.sum(probs) == 0: ## In the case with F ward it is impossible to move patients from other wards to F (but not viceversa)
                relocated[n_patient_type] -= 1
                rejected[n_patient_type] += 1
            else:
                probs_normalized = probs/np.sum(probs) ## Normalize the probabilities to sample from them
                select_ward = choice(wards_available, p=probs_normalized) ## Choose one of the other available wards
      
                ward_time = ward_time_dist[n_patient_type,n] + arrival_times[n]
                patients_in_bed_times = np.append(patients_in_bed_times, ward_time)
                patients_in_bed_type = np.append(patients_in_bed_type, select_ward)
            
        else: ## No beds available
            rejected[n_patient_type] += 1
            
    return accepted, relocated, rejected, patient_types_count, penalty