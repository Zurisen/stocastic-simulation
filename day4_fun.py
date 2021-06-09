#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:28:02 2021

@author: carlos
"""
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from scipy import stats


### Main 1

def exp_crudeMC(a,b,n_uni=100, n_iter=10000):
	""" Crude Monte-Carlo estimator """
	value_array = np.zeros(n_iter)
	for i in range(n_iter):
		U = uniform(a,b,size=n_uni)
		value = (b-a)/n_uni*np.sum(np.exp(U))
		value_array[i] = value
	
	# Statistics over all the iterations to give the most precise value
	value_mean = np.mean(value_array) 
	value_std = np.std(value_array)
	return value_mean, value_std, value_array


def exp_antithetic(a,b,n_uni=100, n_iter=10000):
	""" Antithetic variables estimator """
	value_array = np.zeros(n_iter)
	for i in range(n_iter):
		U = uniform(a,b,size=n_uni)
		value = (b-a)/n_uni* np.sum( (np.exp(U) + np.exp(1-U))/2 )
		value_array[i] = value
	
	# Statistics over all the iterations to give the most precise value
	value_mean = np.mean(value_array) 
	value_std = np.std(value_array)
	return value_mean, value_std, value_array


def exp_cvar(a,b,n_uni=100000):
	""" Control variable estimator """
	value_array = np.zeros(n_uni)
	
	U = uniform(a,b,size=n_uni)
	cov = np.cov(U,np.exp(U))
	mean = np.mean(U)
	c = -cov[0,1]/cov[0,0]
	
	value_array = np.exp(U)+c*(U-mean)
	
	# Statistics over all the iterations to give the most precise value
	value_mean = np.mean(value_array) 
	value_std = np.std(value_array)
	return value_mean, value_std, value_array


def exp_stratified(a,b,n_uni=1000, n_iter=10):
	""" Stratified sampling estimator """
	U = uniform(a,b,size=[n_iter,n_uni])
	foo = np.arange(0,n_iter,1)/n_iter
	foomat = np.zeros((n_iter,n_uni))
	for i in range(n_uni):
		foomat[:,i] = foo
		
	Wi = np.sum( np.exp(foomat+U/n_iter) ,axis=0)/n_iter
	
	# Statistics over all the iterations to give the most precise value
	value_mean = np.mean(Wi) 
	value_std = np.std(Wi)
	return value_mean, value_std, Wi


#### Main 2

def cvar(X,Y):
	cov = np.cov(X,Y)
	meanY = np.mean(Y)
	c = -cov[0,1]/cov[1,1]
	
	value_array = X+c*(Y-meanY)
	
	# Statistics over all the iterations to give the most precise value
	value_mean = np.mean(value_array) 
	value_std = np.std(value_array)
	return value_mean, value_std, value_array

def exp_importance(a,b,lamb,n_uni=1000):
	""" Importance sampling estimator """
	X = uniform(a,b,size=n_uni)
	Y = 1-X
	gY = lamb*np.exp(-lamb*Y)
	hYfY = np.exp(Y)*X
	
	value_array = hYfY/gY
	value_mean = np.mean(value_array)
	value_std = np.std(value_array)
	return value_mean, value_std, value_array


def normal_crudeMC(a,n_uni=10000, n_iter=10000):
	""" Crude Monte-Carlo estimator for X>a"""
	value_array = np.zeros(n_iter)
	for i in range(n_iter):
		N = normal(0,1,size=n_uni)
		value = np.sum(N>a)/n_uni
		value_array[i] = value
	
	# Statistics over all the iterations to give the most precise value
	value_mean = np.mean(value_array) 
	value_std = np.std(value_array)
	return value_mean, value_std, value_array
