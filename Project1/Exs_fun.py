#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:05:28 2021

@author: carlos
"""
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from scipy import stats
from random import sample

def normalize_rows(Pt):
    sum_rows = np.sum(Pt,axis=1)
    Pt = Pt/sum_rows
    return Pt    

def update1_Pt(Pt,P0):
    Pt = np.matmul(P0,Pt)
    Pt = normalize_rows(Pt)
    return Pt


def simulate_death_1(P, n_women, limit_months=240):
    n_states = np.arange(0,P.shape[0])
    women_states_array = np.full((n_women,limit_months), fill_value=np.nan)
    women_months = np.zeros(n_women)
    for women in range(n_women):
        month = 0
        state = 0 ## All women start perfectly healthy
        women_states_array[women,month] = state ## Initialize woman state
        while (month<limit_months-1) & (state<n_states[-1]):
            new_state = choice(n_states, p = P[state]) ## Sample new state
            state = new_state # Update state
            month += 1 
            
            women_states_array[women,month] = state ## Save previous state value
        women_months[women] = month
        
    return women_states_array, women_months

def simulate_death_1_analytical(P, n_women, limit_months=240):
    n_states = np.arange(0,P.shape[0])
    women_states_array = np.full((n_women,limit_months), fill_value=np.nan)
    women_months = np.zeros(n_women)
    for women in range(n_women):
        Pt = P
        month = 0
        state = 0 ## All women start perfectly healthy
        women_states_array[women,month] = state ## Initialize woman state
        while (month<limit_months-1) & (state<n_states[-1]):
            new_state = choice(n_states, p = Pt[state]) ## Sample new state
            state = new_state # Update state
            month += 1 
            Pt = update1_Pt(Pt,P) #Update probabilities
            
            women_states_array[women,month] = state ## Save previous state value
        women_months[women] = month
        
    return women_states_array, women_months

def empirical_lifetime(P, t):
    pi = np.array([1,0,0,0])
    ps = P[:-1,-1]
    Ps = P[:-1,:-1]
    Pst = np.power(Ps,t)
    
    Prob = np.matmul(np.matmul(pi,Pst),ps)
    Mean = np.matmul(np.matmul(pi, np.linalg.inv(np.identity(len(pi))-Ps)), np.ones(len(pi)) )
    
    return Prob, Mean
    
    
    