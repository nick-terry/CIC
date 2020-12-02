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
    
    results = stat.norm.logpdf(x,0,3)
    
    # Use log transform to prevent numerical issues
    log_res = np.sum(results,axis=1)
    
    return np.expand_dims(np.exp(log_res),axis=1)



def runReplicate(seed):
    
    # Create a SimulationEngine for evaluating h(x)
    sim = se.SimulationEngine()
    dataDim = sim.numBranches
    
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
    
    np.random.seed(seed)
    
    # Run the CEM procedure
    
    # Variance of each coordinate in initial GMM
    sigmaSq = 3
    
    # Initial guess for GMM params
    k = 30
    alpha0 = np.ones(shape=(k,))/k
    
    # Randomly intialize the means of the Gaussian mixture components
    mu0 = np.random.multivariate_normal(np.zeros(dataDim),
                                        np.eye(dataDim),
                                        size=k)
    
    # Set covariance matrix to be identity
    sigma0 = sigmaSq * np.repeat(np.eye(dataDim)[None,:,:],k,axis=0)
    initParams = cem.GMMParams(alpha0, mu0, sigma0, dataDim)

    sampleSize = [6000,] + [4000,]*2 
    
    procedure = cem.CEM(initParams,p,h,numIters=len(sampleSize),sampleSize=sampleSize,seed=seed,
                        log=True,verbose=True,covar='homogeneous')
    procedure.run()
    
    # Estimate the failure probability
    rho = procedure.rho()
    
    print('Done with replicate!')
    
    return rho,procedure

if __name__ == '__main__':
    
    np.random.seed(42)
    
    # Use importance sampling to estimate probability of cascading blackout given
    # a random N-2 contingency.
    
    # x = np.random.normal(10,3,size=(5,dataDim))
    # Hx = h(x)
    
    numReps = 28
    
    # Get random seeds for each replication
    seeds = np.ceil(np.random.uniform(0,99999,size=numReps)).astype(int)
    
    # Create multiprocessing pool w/ 28 nodes for Hyak cluster
    with mp.Pool(28) as _pool:
        result = _pool.map_async(runReplicate,
                                  list(seeds),
                                  callback=lambda x : print('Done!'))
        result.wait()
        resultList = result.get()
    # rhoList = []
    # for seed in list(seeds):
    #     rho,ce = runReplicate(seed,dataDim)
    #     rhoList.append(rho)
    
    rhoList = [item[0] for item in resultList]
    # toCsvList = [[rho,] for rho in rhoList]
    toCsvList = [[item[0],item[1].paramsList[-1].k()] for item in resultList]
    
    print('Mean: {}'.format(np.mean(rhoList)))
    print('Std Err: {}'.format(stat.sem(rhoList)))
    # Save the estimates of failure probabilty to csv
    with open('results.csv','w') as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow(['rho','final_k'])
        writer.writerows(toCsvList)
    
