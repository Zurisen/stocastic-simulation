#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 09:57:11 2021

@author: carlos
"""
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from scipy import stats
from day6_fun import *
#%%
M_cost = mat2()

total_cost, route, T = min_cost_overall(M_cost,1,5)


fig, ax = plt.subplots(2,1, figsize=(6,10))
ax[0].plot(total_cost,T)
ax[0].set_ylabel("Temperature")
ax[1].hist(total_cost)
ax[1].set_ylabel("Counts")
ax[1].set_xlabel("Total cost")