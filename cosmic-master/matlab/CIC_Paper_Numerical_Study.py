#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:56:46 2020

@author: nick
"""

import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import csv

# Import cem test version (stored locally, NOT a package!)
import cem

'''
Implement example from Kurtz and Song 2013.
Eq (19) and setup in p. 39.
Table 2 Example
'''

def p(x):
    """
    Compute the true likelihood of x from multivariatenormal distribution

    Parameters
    ----------
    x : numpy array

    Returns
    -------
    likelihood : numpy array

    """
    
    results = stat.multivariate_normal.pdf(x,np.zeros(shape=(2,)),np.eye(2))
    
    return np.expand_dims(results,axis=1)

def h(x):
    '''
    h(x) defined in Kurtz and Song

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    bb = 2
    kk = .1
    ee = 0
    
    h_x = (bb-x[:,1]-kk*(x[:,0]-ee)**2) <= 0
    return np.expand_dims(h_x,axis=1)

def plotGMM(params,q,_ax=None,circle=False):
    x_coords = np.linspace(-10,10,num=1000)
    y_coords = np.linspace(-10,10,num=1000)
    coords_grid = np.transpose([np.tile(x_coords, y_coords.size),
                                np.repeat(y_coords, x_coords.size)])
    q_theta = q(params)
    density_grid = np.reshape(q_theta(coords_grid),(x_coords.size,y_coords.size))
    
    # Draw contours of GMM density
    if _ax is None:
        fig,ax = plt.subplots(1)
    else:
        ax = _ax
        
    contf = ax.contourf(x_coords,y_coords,density_grid,levels=10)
    
    if _ax is None:
        plt.colorbar(contf)
    
    # Mark the means of each Gaussian component
    alpha,mu,sigma = params.get()
    for j in range(params.k()):
        ax.scatter(mu[j,0],mu[j,1],marker='x',color='red')
     
    if circle:
        c1 = plt.Circle((mu[0,0],mu[0,1]),np.linalg.norm(mu)/2,color='red',fill=False)
        ax.add_artist(c1)
        
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(-5,5)
    #ax.set_ylim(2,6)
    
    return ax

def plotStage(paramsList,q,X,s,ax=None):
    
    ax = plotGMM(paramsList[s],q,ax)

    ax.scatter(X[s,:,0],X[s,:,1],s=2,color='yellow',alpha=.2,zorder=1)
    
def plotStages(paramsList,q,X):
    
    numIters = len(paramsList)
    sq = np.ceil(np.sqrt(numIters)).astype(int)
    fig,axes = plt.subplots(sq,sq)
    axes = [ax for sublist in axes for ax in sublist]
    
    for s in range(numIters):
        plotStage(paramsList,q,X,s,axes[s])

def runReplicate(seed):
    
    np.random.seed(seed)
    
    # Run the CEM procedure
    dataDim = 2
    # mu = 0 * np.ones(shape=(numComponents,1))
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

    procedure = cem.CEM(initParams,p,h,numIters=7,sampleSize=1000,seed=seed,log=True)
    procedure.run()
    
    # Estimate the failure probability
    rho = procedure.rho(procedure.X,
                        procedure.rList,
                        procedure.q,
                        procedure.paramsList)
    
    print('Done with replicate!')
    
    return rho,procedure

if __name__ == '__main__':
    
    np.random.seed(420)
    
    numReps = 50
    
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
    # for seed,i in zip(list(seeds),range(numReps)):
    #     rho,ce = runReplicate(seed,i)
    #     rhoList.append(rho)
    
    rhoList = [item[0] for item in resultList]
    #toCsvList = [[rho,] for rho in rhoList]
    toCsvList = [[item[0],item[1].paramsList[-1].k()] for item in resultList]
    
    print('Mean: {}'.format(np.mean(rhoList)))
    print('Std Err: {}'.format(stat.sem(rhoList)))
    # Save the estimates of failure probabilty to csv
    with open('results.csv','w') as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow(['rho','final_k'])
        writer.writerows(toCsvList)
    

    