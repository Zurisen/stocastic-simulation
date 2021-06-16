#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:11:25 2021

@author: carlos
"""
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from scipy import stats
from random import sample
from Exs_fun import *
#%% 
######################
##### EXERCISE 1 #####
######################

P = np.array([[0.9915,0.005,0.0025,0,0.001],[0,0.986,0.005,0.004,0.005],\
              [0,0,0.992,0.003,0.005],[0,0,0,0.991,0.009],[0,0,0,0,1]])

n_women = 1000
#%%
""" Simulate 1000 women Exercise 1"""
women_states, women_months = simulate_death_1(P,n_women)
women_months = women_months[women_months!=(np.max(women_months))]
mean_month = np.mean(women_months)
std_month = np.std(women_months)
## Number of alive and dead women per month
dead = np.cumsum(np.sum(women_states==4, axis=0))
alive = n_women-dead

plt.figure()
plt.plot(dead,label="Dead women")
plt.plot(alive,label="Alive women")
plt.xlabel("Month")
plt.ylabel("Number of women")
plt.legend()

plt.figure()
ci = stats.norm.interval(0.95, loc=mean_month, scale=std_month) # 95% confidence intervals

plt.hist(women_months,bins=20)

plt.title("1000 women simulation Exercise 1")
plt.axvline(x=ci[0], color="black", alpha=0.8,linestyle="--")
plt.axvline(x=ci[1], color="black", alpha=0.8,linestyle="--")
plt.xlabel("Month of death")
plt.ylabel("Number of women")

""" Find in how many women cancer reappears locally """
count_local = np.sum(np.sum(women_states==1,axis=1)!=0)

print("Number of women who suffered reapparance of local cancer: %d out of 1000" %count_local, "(%1.2f%%)" %(count_local/n_women*100))

#%%
""" Simulate 1000 women Exercise 1 analytical"""
women_states_an, women_months_an = simulate_death_1_analytical(P,n_women)
mean_month_an = np.mean(women_months_an)
std_month_an = np.std(women_months_an)

## Number of alive and dead women per month (analytical solution)
dead_an = np.cumsum(np.sum(women_states_an==4, axis=0))
alive_an = n_women-dead_an

plt.figure()
plt.plot(dead_an,label="Dead women")
plt.plot(alive_an,label="Alive women")
plt.xlabel("Month")
plt.ylabel("Number of women")
plt.legend()
n_women
plt.figure()
ci_an = stats.norm.interval(0.95, loc=mean_month_an, scale=std_month_an) # 95% confidence intervals

plt.hist(women_months_an,bins=20)

plt.title("1000 women simulation Exercise 1 analytical result")
plt.axvline(x=ci_an[0], color="black", alpha=0.8,linestyle="--")
plt.axvline(x=ci_an[1], color="black", alpha=0.8,linestyle="--")
plt.xlabel("Month of death")
plt.ylabel("Number of women")

#%% 
######################
##### EXERCISE 2 #####
######################

plt.figure()

""" Distributions of states at t=120 """
women_states_120 = women_states[:,119][~np.isnan(women_states[:,119])]

a = plt.hist(women_states_120,bins=5)

plt.bar([0,1,2,3,4],a[0])
plt.ylabel("Number of women")
plt.xlabel("State of women at month 120")
#%%
""" Check Mean and Probability of healthy women at time t """

t_total = 240
Probs= np.zeros(t_total)

for t in range(t_total):
    Probt, Meant = empirical_lifetime(P, t=t)
    Probs[t] = Probt

plt.figure()
plt.plot(Probs)
plt.title("Probability of a woman to die at different points in time (Probability distr. function)")
plt.xlabel("Month")
plt.ylabel("Probability")

plt.figure()
plt.plot(np.cumsum(Probs))
plt.title("Probability of a woman to be dead at different points in time (Cumulative distr. function)")
plt.xlabel("Month")
plt.ylabel("Probability")

#%%
######################
##### EXERCISE 4 #####
######################

women_states, women_months = simulate_death_1_analytical(P,n_women=5000)
#%%
n_accepted = 1000
accepted_women_states, accepted_women_months = rejection_sampling(P, women_states, women_months, n_accepted=n_accepted)

mean_month = np.mean(accepted_women_months)
std_month = np.std(accepted_women_months)
## Number of alive and dead women per month
dead = np.cumsum(np.sum(accepted_women_states==4, axis=0))
alive = n_accepted-dead

plt.figure()
plt.plot(dead,label="Dead women")
plt.plot(alive,label="Alive women")
plt.xlabel("Month")
plt.ylabel("Number of women")
plt.legend()

plt.figure()
ci = stats.norm.interval(0.95, loc=mean_month, scale=std_month) # 95% confidence intervals

plt.hist(accepted_women_months,bins=20)

plt.title("1000 accepted women simulation Exercise 4")
plt.axvline(x=ci[0], color="black", alpha=0.8,linestyle="--")
plt.axvline(x=ci[1], color="black", alpha=0.8,linestyle="--")
plt.xlabel("Month of death")
plt.ylabel("Number of women")

#%%
######################
##### EXERCISE 5 #####
######################
n_women = 200
limit_months = 350
n_iter = 100


X,Z = control_variate(P, n_women, n_iter, limit_months, method="standard")
#%%
plt.figure()
plt.hist(X, label="X",bins=20)
ci_X = stats.norm.interval(0.95, loc=np.mean(X), scale=np.std(X)) # 95% confidence intervals
plt.axvline(x=ci_X[0], color="blue", alpha=0.8,linestyle="--")
plt.axvline(x=ci_X[1], color="blue", alpha=0.8,linestyle="--")

plt.hist(Z, label="Z",bins=20)
ci_Z = stats.norm.interval(0.95, loc=np.mean(Z), scale=np.std(Z)) # 95% confidence intervals
plt.axvline(x=ci_Z[0], color="orange", alpha=0.8,linestyle="--")
plt.axvline(x=ci_Z[1], color="orange", alpha=0.8,linestyle="--")

plt.xlabel("Fraction of women dying within the first 350 months")
plt.legend()

var_X = np.var(X)
var_Z = np.var(Z)
print("There is a reduction of variance of %1.6f -->" %var_X,"%1.6f" %var_Z, "/%1.2f%%)" %(var_Z/var_X*100))


