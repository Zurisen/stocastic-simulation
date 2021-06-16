#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 13:32:46 2021

@author: carlos
"""
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from scipy import stats
from random import sample

def simulate_death_2(Q, n_women, limit_months=30.5):
    n_states = np.arange(0,Q.shape[0])
    last_states = np.zeros(n_women)
    women_months = np.zeros(n_women)
    for women in range(n_women):
        month = 0
        state = 0 ## All women start perfectly healthy
        last_states[women] = state ## Initialize woman state
        while (month<limit_months) & (state<n_states[-1]):
            month += exponential(scale=-1/Q[state,state], size=1)
            new_state = choice(n_states[n_states!=state], p = -Q[state, Q[state]!=Q[state,state]]/Q[state,state]) ## Sample new state
            state = new_state # Update state    
            
            last_states[women] = state ## Save previous state value
        women_months[women] = month
        
    return last_states, women_months

def simulate_death_2_analytical(Q, n_women, limit_months=30.5):
    n_states = np.arange(0,Q.shape[0])
    last_states = np.zeros(n_women)
    women_months = np.zeros(n_women)
    for women in range(n_women):
        month = 0
        state = 0 ## All women start perfectly healthy
        last_states[women] = state ## Initialize woman state
        while (month<limit_months) & (state<n_states[-1]):
            month += exponential(scale=-1/Q[state,state], size=1)
            new_state = choice(n_states[n_states!=state], p = -Q[state, Q[state]!=Q[state,state]]/Q[state,state]) ## Sample new state
            #month += exponential(scale=1/Q[state,new_state], size=1)
            state = new_state # Update state    
            
            last_states[women] = state ## Save previous state value
        women_months[women] = month
        
    return last_states, women_months