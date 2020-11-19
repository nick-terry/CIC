#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 07:29:55 2020

@author: nick
"""

import scipy.stats as stat
import numpy as np
import time
import csv

# Import cem test version (stored locally, NOT a package!)
import simengine as se

sim = se.SimulationEngine()

def p(x):
    """
    Compute the true likelihood of x from normal distribution

    Parameters
    ----------
    x : numpy array

    Returns
    -------
    likelihood : numpy array

    """
    
    results = stat.norm.logpdf(x,10,3)
    
    # Use log transform to prevent numerical issues
    log_res = np.sum(results,axis=1)
    
    return np.expand_dims(np.exp(log_res),axis=1)

def _runSim(x):
    """
    Helper function to run a simulation of the N-2 contingency 

    Parameters
    ----------
    x : numpy array
        The time-to-failure vector.

    Returns
    -------
    results : tuple
        The results of the N-2 contingency simulation.

    """
    
    x = np.squeeze(x)
    
    # Pull out the branches that failed and simulate.
    failInd = np.where(x==1)[0]
    br1,br2 = int(failInd[0]),int(failInd[1])
    return sim._simulate(br1,br2)

def h(x):
    
    # Convert the time-to-failure vector to a boolean vector of contingencies
    contingencies = ttfToContingency(x)
    
    if len(x.shape) > 1 and x.shape[0] > 1:
        n = x.shape[0]

        results = np.zeros(shape=(n,1))
        times = []
        
        for i in range(n):
            # Take only the first item from the result tuple (blackout boolean)
            start_time = time.time()
            result = _runSim(contingencies[i,:])[0]
            end_time = time.time()
            times.append(end_time-start_time)
            results[i] = result
            
    else:
        start_time = time.time()
        results = np.array(_runSim(contingencies))[0].reshape((1,1))
        end_time = time.time()
        times = [end_time-start_time,]
        
    return results,times

def ttfToContingency(x):
    
    # Use failure times to generate contingency vector
    failInd = np.argsort(x,axis=1)[:,:2].astype(int)
    contingency = np.zeros_like(x)
    np.put_along_axis(contingency, failInd, 1, axis=1)
    
    return contingency
    

if __name__ == '__main__':
    
    np.random.seed(42)
    
    numSamples = 1000
    
    x = np.random.normal(10,3,size=(numSamples,46))
    
    # Benchmark the time to evaluate 10000 simulations
    Hx,times = h(x)
    toCSVList = [[time,output] for time,output in zip(times,Hx.flatten())]

    with open('benchmark.csv','w') as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow(['time','output'])
        writer.writerows(toCSVList)
    
    