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

def rejection_sampling(P, women_states_array, women_months, n_accepted=1000):
    ## Initialize accepted women
    accepted_women_states_array = np.full((n_accepted, women_states_array.shape[1]), fill_value=np.nan)
    accepted_women_months = np.zeros(n_accepted)
    ## Create classes (state of the women) probabilities for rejection sampling
    n_classes = P.shape[0]
    class_prob_distribution = np.zeros(n_classes)
    for n in range(n_classes):    
        class_prob_distribution[n] = np.sum(women_states_array[:,11]==n)/women_states_array.shape[0]
    
    women = 0
    accepted = 0
    rejected = 0
    
    ## Reject Nans at month 12
    wherenot_Nans = np.where(~np.isnan(women_states_array[:,11]))
    rejected += women_states_array.shape[0]-len(wherenot_Nans[0])
    women_states_array = women_states_array[wherenot_Nans[0]]
    
    while (accepted < n_accepted):
        women_state_12 = int(women_states_array[women,11])
        u = uniform(0,1)
        
        if (women_states_array[women,11] not in [0,4]) & (class_prob_distribution[women_state_12]<u):
            accepted_women_states_array[accepted] = women_states_array[women]
            accepted_women_months[accepted] = women_months[women]
            accepted += 1
        else:
            rejected +=1

        women += 1
        #print(accepted)
        if women == women_states_array.shape[0]:
            print("Accepted %d" %accepted, "out of %d expected" %n_accepted )
            raise AssertionError("Not enough accepted women, input more samples!")
        
    
        
    return accepted_women_states_array, accepted_women_months
    
    
def control_variate(P, n_women, n_iter, limit_months, method="standard"):
    X = np.zeros(n_iter) ## We will store here the fraction of dead women per simulation
    Y = np.zeros(n_iter) ## We will store here the mean lifetime after surgery per simulation
    for i in range(n_iter):
        if method=="standard":
            women_states_array, women_months = simulate_death_1(P, n_women, limit_months)
        elif method=="analytical":
            women_states_array, women_months = simulate_death_1_analytical(P, n_women, limit_months)
        else:
            raise NotImplementedError("Wrong method: Choose standard or analytical method")
        X[i] = np.sum(np.isnan(women_states_array[:,-1]))/n_women
        Y[i] = np.mean(women_months)
    
    cov = np.cov(X,Y)
    meanY = np.mean(Y)
    c = -cov[0,1]/cov[1,1]

    Z = X + c*(Y-meanY)
    
    return X, Z
    