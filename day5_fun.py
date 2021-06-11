#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:01:30 2021

@author: carlos
"""
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from scipy import stats
#from math import factorial
from scipy.special import factorial

def metropolis_hastings(A,m,n=10000):
	""" Truncated Poisson - Metropolis hastings"""
	I = np.zeros(n)
	#dtor_index = np.arange(0,m,1)
	#dtor = np.sum(np.power(A,dtor_index)/dtor_index)
	
	for t in range(n-1):
		i = I[t]
		i_ = randint(0,m)
		
		P = (np.power(A,i)/factorial(i))
		P_ = (np.power(A,i_)/factorial(i_))
		
		accept_prob = min(1,P_/P)
		u = uniform(0,1)
		if u <= accept_prob:
			I[t+1] = i_
		else:
			I[t+1] = i

	return I
	
def mixed_metropolis_hastings(A1,A2,m,n=10000, coordinate_wise=False):
	""" Truncated Poisson - Mixed Metropolis hastings"""
	I = np.zeros(n)
	J = np.zeros(n)
	I[0] = randint(0,m)
	J[0] = randint(0,m-I[0])
	if not coordinate_wise:
		for t in range(n-1):
			i = I[t]
			j = J[t]
			i_ = randint(0,m)
			j_ = randint(0,m-i_)
		
			P = np.power(A1,i)*np.power(A2,j) / ( factorial(i)*factorial(j) ) 
			P_ = np.power(A1,i_)*np.power(A2,j_) / ( factorial(i_)*factorial(j_) ) 
		
			accept_prob = min(1,P_/P)
			u = uniform(0,1)
			if u <= accept_prob:
				I[t+1] = i_
				J[t+1] = j_
			else:
				I[t+1] = i
				J[t+1] = j

		return I,J
	
	else:
		for t in range(n-1):
			i = I[t]
			j = J[t]
			i_ = randint(0,m)
			j_ = randint(0,m-i_)
			
			Pi = np.power(A1,i)/factorial(i)
			Pi_ = np.power(A1,i_)/factorial(i_)
			Pj = np.power(A2,j)/factorial(j)
			Pj_ = np.power(A2,j_)/factorial(j_)
			
			accept_probi = min(1,Pi_/Pi)
			accept_probj = min(1,Pj_/Pj)
			u = uniform(0,1)
			
			if u <= accept_probi:
				I[t+1] = i_
			else: 
				I[t+1] = i
			if u <= accept_probj:
				J[t+1] = j_
			else:
				J[t+1] = j
			
		return I,J
		
def gibbs_sampling(A1,A2,m,n=10000):
	I = np.zeros(n)
	J = np.zeros(n)
	I[0] = randint(0,m)
	J[0] = randint(0,m-I[0])
	indx = np.arange(0,m,1)
	cteterm =np.sum( np.power(A1,indx)/factorial(indx) )
	print(cteterm)
	for t in range(n-1):
		if (t%2) == 0:
			I[t+1] = np.power(A1,I[t])/factorial(I[t])*cteterm
			print(np.power(A1,I[t])/factorial(I[t]))
					
		else:
			J[t+1] = np.power(A2,J[t])/factorial(J[t])*cteterm
	return I,J
