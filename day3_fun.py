#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 08:12:09 2021

@author: carlos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from time import time
from numpy.random import *

def hyperexp(p1,p2,l1,l2, size):
	""" Hyperexponential distribution"""
	
	u = random_sample(size)
	index_1 = np.where(u<p1)
	index_2 = np.where(u>=p1)

	u[index_1] = exponential(size=len(index_1[0]), scale=1/l1)
	u[index_2] = exponential(size=len(index_2[0]), scale=1/l2)
	return u


def queue_simulation_merged(n_su, mst, mtbc, n_customers, arrival_dist_type, service_dist_type):
	"""

	Parameters
	----------
	n_su : int
		number of service units.
	mst : int
		mean service time.
	mtbc : int
		mean time between customers.
	n_customers : int
		number of customers.
	arrival_dist_type : char string
		arrival distribution type.
	service_dist_type : char string
		service distribution type.

	Raises
	------
	NotImplementedError
		Error indicating if an arrival or service method has been implemented.

	Returns
	-------
	blocked_customers : int
		number of blocked customers.

	"""
	blocked_customers = 0
	service_unit_times = np.zeros(n_su) # Each unit initialites at time 0 (they are not busy)
	
	arrival_dist_list = {
						"poisson": poisson(size=[n_customers], lam=mtbc), ## Poisson dist
						"erlang": gamma(shape=1,size=[n_customers], scale=mtbc), ## Erlang dist
						"hyperexp": hyperexp(0.8,0.2,0.8333,5, size=n_customers) ## hyperexp
						}


	service_dist_list = {
						"exp": exponential(size=[n_customers], scale=mst), ## Exponential dist
						"constant": np.full(fill_value=20, shape=[n_customers]), ## Constant, the higher the fill value the slower the service is
						"pareto": pareto(1.05, size=[n_customers]), # Standard Pareto distribution with values k=1.05 or k=2.05
						"uniform": uniform(5,20, size=[n_customers]), # Uniform distribution
						"normal": np.abs(normal(10,5, size=[n_customers])) # Normal (fully positive) distribution
						}

	try:
		arrival_time_dist = arrival_dist_list[arrival_dist_type] ## Select service distribution
	except:
		raise NotImplementedError("Not available arrival distribution type")

	try:
		service_time_dist = service_dist_list[service_dist_type] ## Select service distribution
	except:
		raise NotImplementedError("Not available service distribution type")
		
	
	time = np.cumsum(arrival_time_dist) # Sequential times of customer arrivals
	for i in range(n_customers):
		if sum(service_unit_times<time[i]) > 0: # if the arrival time is lower than the service time of any server, it blocks the customer
			index = np.where(service_unit_times<time[i])[0][0] # the next arrival gets into the free servers (we pick the first free)
			service_unit_times[index] = service_time_dist[i] + time[i] # Adds the service unit time to the time the customer has arrived
		else:
			blocked_customers += 1
	return blocked_customers
