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

def beds_simulation(realoc_probs, params, n_patients=5000):
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
    for i in range(n_patients):
        patient_type = randint(0,n_types)
        arrival_time_dist[i] = exponential(size=1, scale=1/arrival_rates[patient_type])
        arrival_types[i] = patient_type
    arrival_times = np.cumsum(arrival_time_dist)
    
    ## Build ward service times
    ward_time_dist = np.zeros((n_types, n_patients))
    for i in range(n_types):
        ward_time_dist[i] = exponential(size=[n_patients], scale = ward_rates[i])
    
    ## Start counter of beds occupation for each ward
    beds_occupied = np.zeros(n_types)
    ## Start time of occupation for each ward
    ward_times = np.zeros(n_types)
    
    ## Start counter for rejected patients
    rejected = 0
    
    for n in range(n_patients):
        n_patient_type = int(arrival_types[n])
        just_freed_beds = np.where(ward_times<arrival_times[n])[0] ## Check beds that has just been freed
        # print("#######")
        # print(bed_capacities)
        # print(beds_occupied)

        if just_freed_beds.shape[0] != 0:
            beds_occupied[just_freed_beds] -= 1 ## Remove patient from the ward
            beds_occupied[beds_occupied<0] = 0 ## To avoid negative occupation values
            
        if (beds_occupied[n_patient_type] < bed_capacities[n_patient_type]):
            beds_occupied[n_patient_type] += 1
            ward_times[n_patient_type] = ward_time_dist[n_patient_type,n] + arrival_times[n]
            
        elif np.sum(beds_occupied<bed_capacities)>0: ## Check if there is any other ward with beds available
            wards_available = np.where(beds_occupied<bed_capacities)[0]
            probs = realoc_probs[n_patient_type,wards_available]
            
            if probs[0] == 0:
                rejected += 1
            else:
                probs_normalized = probs/np.sum(probs) ## Normalize the probabilities to sample from them
                select_ward = choice(wards_available, p=probs_normalized) ## Choose one of the other available wards
                # print(n_patient_type)
                # print("asdfasdf")
                beds_occupied[select_ward] += 1
                ward_times[select_ward] = ward_time_dist[n_patient_type,n] + arrival_times[n]
                # print(beds_occupied)
            
        else: ## No beds available
            rejected += 1
            
    return rejected