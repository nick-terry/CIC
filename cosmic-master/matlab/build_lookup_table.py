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
import simengine as se

def runSim(contingency):
    
    # Create a SimulationEngine for evaluating h(x)
    sim = se.SimulationEngine()
        
    x = np.squeeze(contingency)
    
    # Pull out the branches that failed and simulate.
    failInd = np.where(x==1)[0]
    br1,br2 = int(failInd[0]),int(failInd[1])
    return sim._simulate(br1,br2)

if __name__ == '__main__':
    
    np.random.seed(42)
    
    # enumerate all contingencies and create lookup table for the results
    numBr = 46 
    contingencies = []
    for br1 in range(numBr):
        for br2 in range(br1+1,numBr):
            
            cont = np.zeros((1,numBr))
            cont[0,br1] = 1
            cont[0,br2] = 1
            
            contingencies.append(cont)
    
    # Create multiprocessing pool w/ 28 nodes for Hyak cluster
    with mp.Pool(1) as _pool:
        result = _pool.map_async(runSim,
                                  contingencies,
                                  callback=lambda x : print('Done!'))
        result.wait()
        resultList = result.get()
    
    outageResults = [int(result[0]) for result in resultList]
    # convert to tuples for storing in dict
    contT = [tuple(np.argsort(c.squeeze())[-2:][::-1]) for c in contingencies]
    simDict = dict(zip(contT,outageResults))
    
    # Save the estimates of failure probabilty to csv
    with open('simDict.pck','wb') as f:
        pickle.dump(simDict,f)
