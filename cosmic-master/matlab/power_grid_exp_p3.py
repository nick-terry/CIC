#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 07:29:55 2020
 
@author: nick
"""

import scipy.stats as stat
import scipy.special as spc
import numpy as np
import multiprocessing as mp
import csv
import pickle

# Import cem test version (stored locally, NOT a package!)
import cemSEIS as cem
# import simengine as se

# load lookup table for simulation results
with open('simDict_10.pck','rb') as f:
    hDict = pickle.load(f)

d = 3

def p(x):
    """
    Compute the true log likelihood of x from normal distribution

    Parameters
    ----------
    x : numpy array

    Returns
    -------
    likelihood : numpy array

    """
    
    results = stat.norm.logpdf(x,0,1)
    
    # Use log transform to prevent numerical issues
    log_res = np.sum(results,axis=1)
    
    return np.expand_dims(log_res,axis=1)

def samplingOracle(n):
    """
    Draw n samples from the nominal distribution.

    Parameters
    ----------
    n : integer
        Number of samples to draw.

    Returns
    -------
    x : np array
        The drawn samples

    """

    x = stat.multivariate_normal.rvs(np.zeros((d,)),cov=np.eye(d),size=n)

    return x

def runReplicate(seed):
    
    # Create a SimulationEngine for evaluating h(x)
    #sim = se.SimulationEngine()
    # dataDim = sim.numBranches
    dataDim = d
    
    # the number of active components
    nActive = d
    
    # load lookup table for simulation results
    with open('simDict_10.pck','rb') as f:
        hDict = pickle.load(f)
    
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
        
        result = hDict[tuple(x[:10])]
    
        return result

    def h(x):
        
        # Convert the time-of-failure vector to a boolean vector of contingencies
        contingencies = getContingency(x,w=1)
        
        if len(x.shape) > 1 and x.shape[0] > 1:
            n = x.shape[0]
    
            results = np.zeros(shape=(n,1))
            
            for i in range(n):
 
                result = _runSim(contingencies[i,:])
                results[i] = result
                    
        else:
            results = np.array(_runSim(contingencies)).reshape((1,1))
            
        return results
    
    def getContingency(x,w=.5):
        
        # failed = 1 * (-w/2 <= x) * (x <= w/2)
        failed = 1 * (x<=-w)
        
        contingency = np.zeros((x.shape[0],46))
        shift = 1
        contingency[:,shift:nActive+shift] = failed
    
        return contingency
    
    np.random.seed(seed)
    
    # Run the CEM procedure
    
    # Variance of each coordinate in initial GMM
    sigmaSq = 3
    
    # Initial guess for GMM params
    k = 10
    alpha0 = np.ones(shape=(k,))/k
    
    # Randomly intialize the means of the Gaussian mixture components
    hw=2
    
    mu0 = np.random.uniform(-hw,hw,size=(k,d))
    
    # Variance of each coordinate in initial GMM
    sigmaSq = 3 * hw**2
    
    # Set covariance matrix to be identity
    sigma0 = sigmaSq * np.repeat(np.eye(dataDim)[None,:,:],k,axis=0)

    # Set covariance matrix to be identity
    sigma0 = sigmaSq * np.repeat(np.eye(dataDim)[None,:,:],k,axis=0)
    initParams = cem.GMMParams(alpha0, mu0, sigma0, dataDim)

    sampleSize = [8000,] + [2000,]*4 + [4000]
    # sampleSize = [1000,]

    procedure = cem.CEMSEIS(initParams,p,samplingOracle,h,numIters=len(sampleSize),sampleSize=sampleSize,seed=seed,
                        log=True,verbose=True,covar='homogeneous',allowSingular=True)
    procedure.run()
    
    # Estimate the failure probability
    rho = procedure.rho()
    k = procedure.paramsList[-1].k()
    
    print('Done with replicate!')
    
    return rho,k

if __name__ == '__main__':
    
    np.random.seed(420)
    
    # Use importance sampling to estimate probability of cascading blackout given
    # a random N-2 contingency. 
    
    # x = np.random.normal(10,3,size=(5,dataDim))
    # Hx = h(x)
    
    numReps = 10
    
    # Get random seeds for each replication
    seeds = np.ceil(np.random.uniform(0,99999,size=numReps)).astype(int)
    
    # Create multiprocessing pool w/ 28 nodes for Hyak cluster
    # with mp.Pool(28) as _pool:
    #     result = _pool.map_async(runReplicate,
    #                               list(seeds),
    #                               callback=lambda x : print('Done!'))
    #     result.wait()
    #     resultList = result.get()
    
    rhoList = []
    for i,seed in enumerate(list(seeds)):
        rho,ce = runReplicate(seed)
        rhoList.append(rho)
    
    toCsvList = [[rho,] for rho in rhoList]
    # rhoList = [item[0] for item in resultList]
    # toCsvList = [[item[0],item[1]] for item in resultList]
    
    print('Mean: {}'.format(np.mean(rhoList)))
    print('Std Err: {}'.format(np.std(rhoList)))
    # Save the estimates of failure probabilty to csv
    # with open('results_power_p3.csv','w') as f:
    #     writer = csv.writer(f)
    #     # Header row
    #     writer.writerow(['rho','final_k'])
    #     writer.writerows(toCsvList)
    
