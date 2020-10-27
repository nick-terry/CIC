#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 07:29:55 2020

@author: nick
"""

import scipy.stats as stat
import scipy.integrate as intgr
import numpy as np
import logging
import datetime as dttm

# Import cem test version (stored locally, NOT a package!)
import cem
import simengine as se

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
    
    results = stat.norm.pdf(x,10,3)
    
    # Use log transform to prevent numerical issues
    log_res = np.sum(np.log(results),axis=1)
    results = np.exp(log_res)
    
    return np.expand_dims(results,axis=1)

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
        
        for i in range(n):
            # Take only the first item from the result tuple (blackout boolean)
            result = _runSim(contingencies[i,:])[0]
            results[i] = result
            
    else:
        results = np.array(_runSim(contingencies))[0].reshape((1,1))
        
    return results

def ttfToContingency(x):
    
    # Use failure times to generate contingency vector
    failInd = np.argsort(x,axis=1)[:,:2].astype(int)
    contingency = np.zeros_like(x)
    np.put_along_axis(contingency, failInd, 1, axis=1)
    
    return contingency

if __name__ == '__main__':
    
    np.random.seed(42)
    
    # Use importance sampling to estimate probability of cascading blackout given
    # a random N-2 contingency.
    
    # Create a SimulationEngine for evaluating h(x)
    sim = se.SimulationEngine()
    numComponents = sim.numBranches
    
    # Mean time to failure for components: start with 10
    mu = 10 * np.ones(shape=(numComponents,1))
    sigma = 3
    
    # Number of different GMM intializations to use
    numGMM = 1
    
    initGMMList = []
    for i in range(numGMM):
        # Initial guess for GMM params
        k = 10
        alpha0 = np.ones(shape=(k,))/k
        
        # Randomly intialize the means of the Gaussian mixture components
        mu0 = np.random.multivariate_normal(np.zeros(numComponents),
                                            np.eye(numComponents),
                                            size=k)
        
        # mu0 = np.random.uniform(-5,5,size=(k,numComponents))
        # mu0 = np.array([[-3,2],
        #                 [3,-1]])
        # Set covariance matrix to be identity
        sigma0 = sigma * np.repeat(np.eye(numComponents)[None,:,:],k,axis=0)
        params = cem.GMMParams(alpha0, mu0, sigma0, numComponents)
        
        initGMMList.append(params)
    
    finalCrossEntropyArr = np.zeros(shape=(numGMM,1))
    trialList = []
    # For each initial GMM, create a CEM and run the procedure
    for i,params in enumerate(initGMMList):
        print('Beginning Trial {}'.format(i))
        
        trial = cem.CEM(params,p,h,numIters=5,numSamples=5)
        trial.run()
        cicList,paramsList,X = trial.getResults()
        
        finalCrossEntropyArr[i] = cicList[-1]
        trialList.append(trial)
        
    # Choose the trial with the smallest ending cross-entropy
    bestTrialInd = np.argmin(finalCrossEntropyArr)
    trialList[bestTrialInd].writeResults('power_grid_cem.pck')
    
