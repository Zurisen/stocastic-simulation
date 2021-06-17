#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 09:53:28 2021

@author: carlos
"""
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from scipy import stats
from random import sample

def simulate_death_3(Q, n_women, doctor_visit=48, limit_months=1200):
    n_states = np.arange(0,Q.shape[0])
    last_states = np.zeros(n_women)
    women_months = np.zeros(n_women)
    maximum_visits = int(np.round(limit_months/doctor_visit)) ## Maximum visits the patients can make supposing an specific limit_months of lifetime
    doctor_register = np.full((n_women,maximum_visits ), fill_value=np.nan) # Store the state of every woman every time they go to the doctor 
    for women in range(n_women):
        month = 0 ## Counter to track the number of months passed
        state = 0 ## All women start perfectly healthy
        visit = 0 ## Counter to track the number of total visits of the patient to the doctor
        last_states[women] = state ## Initialize woman state
        doctor_register[women,visit] = state ## first visit
        visit += 1
        #print("########")
        while (month<limit_months) & (state<n_states[-1]):
            month += exponential(scale=-1/Q[state,state], size=1)
            new_state = choice(n_states[n_states!=state], p = -Q[state, Q[state]!=Q[state,state]]/Q[state,state]) ## Sample new state

            visit_condition = month-doctor_visit*(visit+1) ## Only visits the doctor if the minimum of "doctor_visit" months has passed
            
            if (visit_condition>0) & (state!=4):

                same_state_visits = visit + np.arange(0, int(np.round(visit_condition/doctor_visit)) ) ## Within the time frame of the patient
                #-- being in an specific state it can go to the doctor serveral times (several visits where the patient is in the same state)
                
                if same_state_visits.shape[0] == 0: ## Debugging option to avoid an empty same_state_visits array
                    same_state_visits = np.array([visit]) 
                #print("same_state_visits: ", same_state_visits)
                if same_state_visits.shape[0]>=(maximum_visits-visit): ## Debugging option to prevent an excesively long same_state_visits array
                    
                    same_state_visits = same_state_visits[0:(maximum_visits-visit)]
                    #print("same_state_visits: ", same_state_visits)
                
                if same_state_visits.shape[0] == 0: ## Debugging option to avoid an empty same_state_visits array
                    same_state_visits = np.array([visit]) 
                    
                visit = same_state_visits[-1]+1 ## Counter to track the number of total visits of the patient to the doctor
                #print("State visit: ", state)
                doctor_register[women,same_state_visits] = state ## Register the patient's state in the doctor visits
            #else:
                #print("State without visit: ", state)
                
            state = new_state # Update state 
            
            last_states[women] = state ## Save previous state value
        women_months[women] = month
        
    return doctor_register, last_states, women_months