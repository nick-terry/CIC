#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 07:29:55 2020

@author: nick
"""

import scipy.stats as stat
import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt

from cem import q as getGmmPDF

# Import cem test version (stored locally, NOT a package!)
import cemSEIS as cem
# import simengine as se

d = 2

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

def plotGMM(params,q,_ax=None,circle=False,hw=20):
    coords = np.linspace(-hw,hw,num=1000)
    coords_grid = np.transpose([np.tile(coords, coords.size),
                                np.repeat(coords, coords.size)])
    q_theta = q(params)
    density_grid = np.reshape(q_theta(coords_grid),(coords.size,coords.size))
    
    # Draw contours of GMM density
    if _ax is None:
        fig,ax = plt.subplots(1)
    else:
        ax = _ax
        
    contf = ax.contourf(coords,coords,density_grid,levels=10,cmap='bone')
    
    if _ax is None:
        plt.colorbar(contf)
    
    if type(params)!=list:
        
        # Mark the means of each Gaussian component
        alpha,mu,sigma = params.get()
        for j in range(params.k()):
            ax.scatter(mu[j,0],mu[j,1],marker='x',color='red')
         
        if circle:
            c1 = plt.Circle((mu[0,0],mu[0,1]),np.linalg.norm(mu)/2,color='red',fill=False)
            ax.add_artist(c1)
        
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(-hw,hw)
    ax.set_ylim(-hw,hw)
    
    return ax

def runReplicate(seed):
    
    # Create a SimulationEngine for evaluating h(x)
    #sim = se.SimulationEngine()
    # dataDim = sim.numBranches
    dataDim = d
    R = .1
    
    def h(x):
        
        # failed = np.product(1 * (x > 1.5), axis=1, keepdims=True)
        A = np.eye(dataDim)
        failed = np.sum((x @ A) * x,axis=1,keepdims=True)<= R
        
        return failed
    
    np.random.seed(seed)
    
    # Run the CEM procedure
    
    # Variance of each coordinate in initial GMM
    sigmaSq = 3
    
    # Initial guess for GMM params
    k = 10
    alpha0 = np.ones(shape=(k,))/k
    
    # Randomly intialize the means of the Gaussian mixture components
    hw=5
    
    mu0 = np.random.uniform(-hw,hw,size=(k,d))
    
    # Variance of each coordinate in initial GMM
    sigmaSq = 3 * hw**2
    
    # Set covariance matrix to be identity
    sigma0 = sigmaSq * np.repeat(np.eye(dataDim)[None,:,:],k,axis=0)

    initParams = cem.GMMParams(alpha0, mu0, sigma0, dataDim)

    sampleSize = [4000,] + [1000,]*3 + [2000]
    
    procedure = cem.CEMSEIS(initParams,p,samplingOracle,h,
                            numIters=len(sampleSize),
                            sampleSize=sampleSize,
                            seed=seed,
                            log=True,
                            verbose=True,
                            covar='homogeneous',
                            alpha=.1)
    procedure.run()
    
    # Estimate the failure probability
    rho = procedure.rho()
    k = procedure.paramsList[-1].k()
    
    print('Done with replicate!')
    
    return rho,k,procedure.paramsList[-1],procedure
    
def plotGMM(params,q,_ax=None,circle=False):
    coords = np.linspace(-5,5,num=1000)
    coords_grid = np.transpose([np.tile(coords, coords.size),
                                np.repeat(coords, coords.size)])

    q_theta = q(params)
    density_grid = np.reshape(q_theta(coords_grid),(coords.size,coords.size))
    
    # Draw contours of GMM density
    if _ax is None:
        fig,ax = plt.subplots(1)
    else:
        ax = _ax
        
    contf = ax.contourf(coords,coords,density_grid,levels=10,cmap='bone')
    
    if _ax is None:
        plt.colorbar(contf)
    
    # # Mark the means of each Gaussian component
    # alpha,mu,sigma = params.get()
    # for j in range(params.k()):
    #     ax.scatter(mu[j,0],mu[j,1],marker='x',color='red')
     
    # if circle:
    #     c1 = plt.Circle((mu[0,0],mu[0,1]),np.linalg.norm(mu)/2,color='red',fill=False)
    #     ax.add_artist(c1)
        
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_0$')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    
    return ax

def plotDensity(densityFn,_ax=None):
    coords = np.linspace(-3,3,num=1000)
    coords_grid = np.transpose([np.tile(coords, coords.size),
                                np.repeat(coords, coords.size)])

    density_grid = np.reshape(densityFn(coords_grid),(coords.size,coords.size))
    
    # Draw contours of GMM density
    if _ax is None:
        fig,ax = plt.subplots(1)
    else:
        ax = _ax
        
    contf = ax.contourf(coords,coords,density_grid,levels=10,cmap='bone')
    
    if _ax is None:
        plt.colorbar(contf)
    
    # # Mark the means of each Gaussian component
    # alpha,mu,sigma = params.get()
    # for j in range(params.k()):
    #     ax.scatter(mu[j,0],mu[j,1],marker='x',color='red')
     
    # if circle:
    #     c1 = plt.Circle((mu[0,0],mu[0,1]),np.linalg.norm(mu)/2,color='red',fill=False)
    #     ax.add_artist(c1)
        
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_0$')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)


if __name__ == '__main__':
    
    np.random.seed(420)
    
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
    for seed in list(seeds):
        rho,ce,params,procedure = runReplicate(seed)
        rhoList.append(rho)
    
    # toCsvList = [[rho,] for rho in rhoList]
    # rhoList = [item[0] for item in resultList]
    # toCsvList = [[item[0],item[1]] for item in resultList]
    
    fig,axes = plt.subplots(1,2)
    plotGMM(params, getGmmPDF, axes[0])
    
    def ISdensity(x):
        failed = 1 * (np.linalg.norm(x,axis=1,ord=2)**2 <= .1)
        
        return stat.multivariate_normal.pdf(x,np.zeros(2),np.eye(2)) * failed
 
    plotDensity(ISdensity,axes[1])
    
    mean = np.mean(rhoList)
    stdErr = stat.sem(rhoList)
    hw = 1.96 * stdErr / np.sqrt(numReps)
    
    print('Mean: {}'.format(np.mean(rhoList)))
    print('Std Err: {}'.format(stat.sem(rhoList)))
    
    print('95% CI for rho_bar: [{:.5f},{:.5f}]'.format(mean-hw,mean+hw))
    # Save the estimates of failure probabilty to csv
    # with open('bias_results_p2.csv','w') as f:
    #     writer = csv.writer(f)
    #     # Header row
    #     writer.writerow(['rho','final_k'])
    #     writer.writerows(toCsvList)
    
