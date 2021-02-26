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
import json

# Import cem test version (stored locally, NOT a package!)
import cem
# import simengine as se

# load lookup table for simulation results
with open('simDict_10.pck','rb') as f:
    hDict = pickle.load(f)

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

def runReplicate(seed):
    
    # Create a SimulationEngine for evaluating h(x)
    #sim = se.SimulationEngine()
    # dataDim = sim.numBranches
    dataDim = 3
    
    def h(x):
        
        failed = np.product(1 * (x > 2), axis=1, keepdims=True)
        
        return failed
    
    np.random.seed(seed)
    
    # Run the CEM procedure
    
    # Variance of each coordinate in initial GMM
    sigmaSq = 3
    
    # Initial guess for GMM params
    k = 10
    alpha0 = np.ones(shape=(k,))/k
    
    # Randomly intialize the means of the Gaussian mixture components
    mu0 = np.random.multivariate_normal(np.zeros(dataDim),
                                        np.eye(dataDim),
                                        size=k)
    
    # Set covariance matrix to be identity
    sigma0 = sigmaSq * np.repeat(np.eye(dataDim)[None,:,:],k,axis=0)
    initParams = cem.GMMParams(alpha0, mu0, sigma0, dataDim)

    # sampleSize = [8000,] + [2000,]*4 + [4000]
    sampleSize = [1000,]
    
    procedure = cem.CEM(initParams,p,h,numIters=len(sampleSize),sampleSize=sampleSize,seed=seed,
                        log=True,verbose=True,covar='homogeneous',allowSingular=True)
    procedure.run()
    
    # Estimate the failure probability
    rho = procedure.rho()
    k = procedure.paramsList[-1].k()
    
    print('Done with replicate!')
    
    return rho,k,np.concatenate(procedure.Hx_WxList,axis=0)

def writeJson(rhoList,likelihoodList):
    
    # Create dictionary for each replication
    toPckList = []
    for rep in range(numReps):
        
        # Create a json representation of the replication.
        repDict = {'rep':rep,'rho':float(rhoList[rep]),
                   'likelihood_ratios':likelihoodList[rep].astype(float).tolist()}
        toPckList.append(repDict)
    
    # Write to pickle file
    with open('replication_data.json', 'w') as f:
        json.dump(toPckList, f)

if __name__ == '__main__':
    
    np.random.seed(123)
    
    # Use importance sampling to estimate probability of cascading blackout given
    # a random N-2 contingency.
    
    # x = np.random.normal(10,3,size=(5,dataDim))
    # Hx = h(x)
    
    numReps = 2
    
    # Get random seeds for each replication
    seeds = np.ceil(np.random.uniform(0,99999,size=numReps)).astype(int)
    
    # # Create multiprocessing pool w/ 28 nodes for Hyak cluster
    # with mp.Pool(28) as _pool:
    #     result = _pool.map_async(runReplicate,
    #                               list(seeds),
    #                               callback=lambda x : print('Done!'))
    #     result.wait()
    #     resultList = result.get()
    rhoList = []
    likelihoodList = []
    for seed in list(seeds):
        rho,ce,likelihoods = runReplicate(seed)
        rhoList.append(rho)
        likelihoodList.append(likelihoods)
    
    writeJson(rhoList,likelihoodList)
    
    toCsvList = [[rho,] for rho in rhoList] 
    
    # rhoList = [item[0] for item in resultList]
    # toCsvList = [[item[0],item[1]] for item in resultList]
    
    print('Mean: {}'.format(np.mean(rhoList)))
    print('Std Err: {}'.format(stat.sem(rhoList)))
    # Save the estimates of failure probabilty to csv
    with open('bias_results.csv','w') as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow(['rho','final_k'])
        writer.writerows(toCsvList)
    
