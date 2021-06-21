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

#%%
realocation_probs = np.array([[0,0.05,0.1,0.05,0.8], [0.2,0,0.5,0.15,0.15], [0.3,0.2,0,0.2,0.3], [0.35,0.3,0.05,0,0.3], [0.2,0.1,0.6,0.1,0]])
parameters = np.array([[55,14.5,2.9,7],[40,11.0,4.0,5],[30,8.0,4.5,2],[20,6.5,1.4,10],[20,5.0,3.9,5]])

rejected = beds_simulation(realocation_probs, parameters, n_patients=1000)

print(rejected)

#%%

realocation_probs = np.array([[0,0.05,0.1,0.05,0.8, 0], [0.2,0,0.5,0.15,0.15,0], [0.3,0.2,0,0.2,0.3,0],
                              [0.35,0.3,0.05,0,0.3,0], [0.2,0.1,0.6,0.1,0,0],[0.2,0.2,0.2,0.2,0.2,0]])
parameters = np.array([[55,14.5,2.9,7],[40,11.0,4.0,5],[30,8.0,4.5,2],[20,6.5,1.4,10],[20,5.0,3.9,5], [20, 13.0, 2.2, 4]])

rejected = beds_simulation(realocation_probs, parameters, n_patients=1000)

print(rejected)