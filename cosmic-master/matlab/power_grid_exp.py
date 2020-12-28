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
import cem
# import simengine as se

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
    
    results = stat.norm.logpdf(x,0,1)
    
    # Use log transform to prevent numerical issues
    log_res = np.sum(results,axis=1)
    
    return np.expand_dims(np.exp(log_res),axis=1)

def runReplicate(seed):
    
    # Create a SimulationEngine for evaluating h(x)
    # sim = se.SimulationEngine()
    # dataDim = sim.numBranches
    dataDim = 46
    
    # load lookup table for simulation results
    with open('simDict.pck','rb') as f:
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
        
        # Pull out the branches that failed and simulate.
        failInd = np.where(x==1)[0]
        br1,br2 = int(failInd[0]),int(failInd[1])
        
        failTup = (br1,br2)
        if failTup in hDict:
            result =  hDict[failTup]
        else:
            result = hDict[failTup[::-1]]
    
        return result

    def h(x):
        
        # Convert the time-to-failure vector to a boolean vector of contingencies
        contingencies = ttfToContingency(x)
        
        if len(x.shape) > 1 and x.shape[0] > 1:
            n = x.shape[0]
    
            results = np.zeros(shape=(n,1))
            
            for i in range(n):
               
                # check if a repair happened
                if np.sum(contingencies[i,:])==0:
                    result = 0
                else:    
                    result = _runSim(contingencies[i,:])
                results[i] = result
                    
        else:
            results = np.array(_runSim(contingencies)).reshape((1,1))
            
        return results
    
    def ttfToContingency(x):
        
        # note: the scale param is what is usually called the rate param for the exp dist
        rate = 5
        # get time to repair 1st failure
        # r = stat.expon.rvs(loc=0,scale=rate,size=x.shape[0])
        rTime = 1/rate
        
        # see if the first failure is fixed before second failure
        firstTwo = np.sort(x,axis=1)[:,:2]
        repaired = np.abs(firstTwo[:,1]-firstTwo[:,0]) > rTime
        
        contingency = np.zeros_like(x)
    
        # Use failure times to generate contingency vector
        failInd = np.argsort(x,axis=1)[:,:2].astype(int)
        np.put_along_axis(contingency, failInd, 1, axis=1)
    
        # set repaired samples to zero
        contingency[repaired] = np.zeros(x.shape[1])
    
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

    sampleSize = [11000,] + [5000,]*4 + [10000]
    # sampleSize = [1000,]
    
    procedure = cem.CEM(initParams,p,h,numIters=len(sampleSize),sampleSize=sampleSize,seed=seed,
                        log=True,verbose=True,covar='full')
    procedure.run()
    
    # Estimate the failure probability
    rho = procedure.rho()
    k = procedure.paramsList[-1].k()
    
    print('Done with replicate!')
    
    return rho,k

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
    #     rho,ce = runReplicate(seed)
    #     rhoList.append(rho)
    
    # toCsvList = [[rho,] for rho in rhoList]
    rhoList = [item[0] for item in resultList]
    toCsvList = [[item[0],item[1]] for item in resultList]
    
    print('Mean: {}'.format(np.mean(rhoList)))
    print('Std Err: {}'.format(stat.sem(rhoList)))
    # Save the estimates of failure probabilty to csv
    with open('results.csv','w') as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow(['rho','final_k'])
        writer.writerows(toCsvList)
    
