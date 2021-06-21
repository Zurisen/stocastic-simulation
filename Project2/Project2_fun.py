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
        

def beds_simulation(realoc_probs, params, urgency_weight = False, ward_distribution = "exp", n_patients=5000):
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
    for i in range(n_patients):
        patient_type = randint(0,n_types)
        patient_types_count[patient_type] +=1
        
        arrival_time_dist[i] = exponential(size=1, scale=1/arrival_rates[patient_type])
        arrival_types[i] = patient_type
    arrival_times = np.cumsum(arrival_time_dist)
    
    ## Build ward service times
    ward_time_dist = np.zeros((n_types, n_patients))
    for i in range(n_types):
        #ward_time_dist[i] = exponential(size=[n_patients], scale = ward_rates[i])
        ward_time_dist[i] = ward_sample(distr_type=ward_distribution, size=n_patients, scale=ward_rates[i])
    
    ## Start counter of beds occupation for each ward
    beds_occupied = np.zeros(n_types)
    ## Start time of occupation for each ward
    ward_times = np.zeros(n_types)
    
    ## Start counter for rejected patients
    rejected = np.zeros(n_types)
    ## Start counter for relocated patients
    relocated = np.zeros(n_types)
    ## Start counter for patients alocated in their correct ward
    accepted = np.zeros(n_types)
    
    ## Count total penalty (we try to minimize)
    penalty = 0
    
    for n in range(n_patients):
        n_patient_type = int(arrival_types[n]) ## keep track of the patient type in this iteration
        just_freed_beds = np.where(ward_times<arrival_times[n])[0] ## Check beds that has just been freed

        if just_freed_beds.shape[0] != 0:
            beds_occupied[just_freed_beds] -= 1 ## Remove patient from the ward
            beds_occupied[beds_occupied<0] = 0 ## To avoid negative occupation values
            
        if (beds_occupied[n_patient_type] < bed_capacities[n_patient_type]):
            accepted[n_patient_type] += 1
            beds_occupied[n_patient_type] += 1
            ward_times[n_patient_type] = ward_time_dist[n_patient_type,n] + arrival_times[n]
            
        elif np.sum(beds_occupied<bed_capacities)>0: ## Check if there is any other ward with beds available
            relocated[n_patient_type] += 1
            penalty += urgency_pts[n_patient_type]
            wards_available = np.where(beds_occupied<bed_capacities)[0]
            probs = realoc_probs[n_patient_type,wards_available]
            
            if np.sum(probs) == 0:
                rejected[n_patient_type] += 1
            else:
                probs_normalized = probs/np.sum(probs) ## Normalize the probabilities to sample from them
                if urgency_weight:
                    probs_normalized[:] = 0
                    min_urgency = np.where(urgency_pts[wards_available] == np.min(urgency_pts[wards_available]))[0]
                    probs_normalized[min_urgency] = 1
                    probs_normalized = probs_normalized/np.sum(probs_normalized)
                
                select_ward = choice(wards_available, p=probs_normalized) ## Choose one of the other available wards
                beds_occupied[select_ward] += 1
                ward_times[select_ward] = ward_time_dist[n_patient_type,n] + arrival_times[n]
            
        else: ## No beds available
            rejected[n_patient_type] += 1
            
    return accepted, relocated, rejected, patient_types_count, penalty