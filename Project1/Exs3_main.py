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
from Exs1_fun import *
from Exs2_fun import *
from Exs3_fun import *
from scipy.linalg import expm
#%%
######################
##### EXERCISE 12 ####
######################
Q = np.array([[-0.00475, 0.0025, 0.00125, 0, 0.001],
              [0,-0.007,0,0.002,0.005],
              [0,0,-0.008,0.003,0.005],
              [0,0,0,-0.009,0.009],
              [0,0,0,0,0]])
n_women = 1000
doctor_visit = 48

doctor_register, last_states, women_months = simulate_death_3(Q, n_women, doctor_visit=doctor_visit, limit_months=1200)

## Check the doctor visit time series of 8 random patients
n_time_series = 50
rand_index = randint(0,n_women,n_time_series)
time_series = doctor_register[rand_index]
visits_timespan = np.arange(0,time_series.shape[1])*doctor_visit

plt.figure()

for i in range(n_time_series):
    color_rand = uniform(0,0.7)
    plt.plot(visits_timespan,time_series[i]+uniform(-0.35,0.35), label=r"Patient "+str(i), 
             color=(color_rand,color_rand,color_rand),marker="o", markersize=2.5)


plt.yticks(np.arange(5), ["Healthy", "Local", "Metastasis", "Local+\n Metastasis", "Dead"])
plt.xlabel("Months")
plt.axhspan(-0.5,0.5,alpha=0.3,color="green")
plt.axhspan(0.5,1.5,alpha=0.3,color="yellow")
plt.axhspan(1.5,2.5,alpha=0.3,color="orange")
plt.axhspan(2.5,3.5,alpha=0.3,color="red")  
plt.axhspan(3.5,4.5,alpha=0.5,color="red")  
plt.title("Evolution of women in every doctor visit")  
plt.ylim(-0.5,4.5)

#%%
Q = np.array([[-0.00475, 0.0025, 0.00125, 0, 0.001],
              [0,-0.007,0,0.002,0.005],
              [0,0,-0.008,0.003,0.005],
              [0,0,0,-0.009,0.009],
              [0,0,0,0,0]])
n_women = 1000
doctor_visit = 48

## We create a simulation nowing Q to get some empirical values, then get the N and S of that
## (supposely given) to run the following simulations
N, S = simulate_death_3_task13(Q, n_women, doctor_visit=48, limit_months=1200)
Q0 = update_Q(N, S)

Qk_, n_iters = converge_Q(Q0, n_women, doctor_visit=48, limit_months=1200)