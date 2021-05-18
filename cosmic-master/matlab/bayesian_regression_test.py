#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 07:29:55 2020

@author: nick
"""

import scipy.stats as stat
import numpy as np
import multiprocessing as mp
import csv
import matplotlib.pyplot as plt

# Import cem test version (stored locally, NOT a package!)
import cemSEIS as cem
from cem import q as getGmmPDF
from cem import getAverageDensityFn as getAvgGmmPDF
# import simengine as se

def h_mu_sigma(theta,mu,sigma):
    """
    Compute the prior density at the parameter values p

    Parameters
    ----------
    theta : numpy array
        Parameters for which the prior is evaluated.
    mu : numpy array
        Mean of prior density for regression coefficients.
    sigma : numpy array
        Covar matrix of prior density for regression coefficients.

    Returns
    -------
    the prior density

    """
    
    '''
    beta : numpy array
        The regression coefficients at which the prior is evaluated.
    logSigma : numpy array
        The log of the observation noise level at which the prior is evaluated.
    '''
    beta,logSigma = theta[:,:-1],theta[:,-1]
    density_beta = stat.multivariate_normal.logpdf(beta,mean=mu,cov=sigma)
    density = np.exp(density_beta + stat.norm.logpdf(logSigma,0,np.sqrt(2)))
    
    return np.expand_dims(density,axis=1)

def p_x(theta,X):
    """
    Compute the likelihood of theta give the data X

    Parameters
    ----------
    theta : numpy array
        Parameters for which likelihood is computed
    X : numpy array
        Predictors/response data used to compute likelihood.

    Returns
    -------
    likelihood : numpy array 
        The likelihood of theta.

    """
    
    
    '''
    beta : numpy array
        The regression coefficients for which the likelihood is being computed.
    sigma : numpy array
        The noise variance for which the likelihood is being computed.
    '''
    beta,logSigma = theta[:,:-1],theta[:,-1]
    sigma = np.exp(logSigma)
    log_results = np.zeros((beta.shape[0],))
    _X,_Y = X[:,:-1].astype(np.float128),X[:,-1].astype(np.float128)
    
    for i in range(beta.shape[0]):
        log_results[i] = np.sum(stat.norm.logpdf(_Y,_X @ beta[i],sigma[i]))
    
    return np.expand_dims(log_results,axis=1)

def samplingOracle_mu_sigma(n,mu,sigma):
    """
    Draw n samples from the prior distribution.

    Parameters
    ----------
    n : integer
        Number of samples to draw.
    mu : numpy array
        mean of prior density for beta
    sigma : numpy array
        covar matrix of prior density for beta

    Returns
    -------
    x : np array
        The drawn samples

    """
    draws = np.zeros((n,mu.size+1))
    
    beta = stat.multivariate_normal.rvs(mu,sigma,size=n)
    logSigma = stat.norm.rvs(0,np.sqrt(2),size=n)
    
    draws[:,:-1] = beta
    draws[:,-1] = logSigma
    
    if len(draws.shape)>1:
        return draws
    else:
        return np.expand_dims(draws,axis=0)


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
    
    # Mark the means of each Gaussian component
    alpha,mu,sigma = params.get()
    for j in range(params.k()):
        ax.scatter(mu[j,0],mu[j,1],marker='x',color='red')
     
    if circle:
        c1 = plt.Circle((mu[0,0],mu[0,1]),np.linalg.norm(mu)/2,color='red',fill=False)
        ax.add_artist(c1)
        
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    
    return ax

def plotMVN(mu,sigma,_ax=None,circle=False):
    coords = np.linspace(-5,5,num=1000)
    coords_grid = np.transpose([np.tile(coords, coords.size),
                                np.repeat(coords, coords.size)])
    
    q_theta = lambda x : stat.multivariate_normal.pdf(x,mu,sigma)
    density_grid = np.reshape(q_theta(coords_grid),(coords.size,coords.size))
    
    # Draw contours of GMM density
    if _ax is None:
        fig,ax = plt.subplots(1)
    else:
        ax = _ax
        
    contf = ax.contourf(coords,coords,density_grid,levels=10,cmap='bone')
    
    if _ax is None:
        plt.colorbar(contf)
     
    if circle:
        c1 = plt.Circle((mu[0,0],mu[0,1]),np.linalg.norm(mu)/2,color='red',fill=False)
        ax.add_artist(c1)
        
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    
    return ax

def getPosterior(mu_prior,sigma_prior,sigma,X):
    
    n = X.shape[0]
    xBar = np.mean(X,axis=0)

    sigma_posterior = np.linalg.inv(np.linalg.inv(sigma_prior)+n*np.linalg.inv(sigma))
    mu_posterior = sigma_posterior@(np.linalg.inv(sigma_prior) @ mu_prior + n*np.linalg.inv(sigma)@xBar)
    
    return mu_posterior,sigma_posterior

def runReplicate(seed,mu_prior,sigma_prior,X):
    
    # Create a SimulationEngine for evaluating h(x)
    #sim = se.SimulationEngine()
    # dataDim = sim.numBranches
    dataDim = 3
    
    np.random.seed(seed)
    
    # Run the CEM procedure
    
    def h(theta): 
        return h_mu_sigma(theta, mu_prior, sigma_prior)
    
    def p(theta):
        return p_x(theta,X)
    
    def samplingOracle(n):
        return samplingOracle_mu_sigma(n, mu_prior, sigma_prior)
    
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

    sampleSize = [200,] + [50,]*4 + [100]
    # sampleSize = [1000,]
    
    procedure = cem.CEMSEIS(initParams,p,samplingOracle,h,
                            numIters=len(sampleSize),
                            sampleSize=sampleSize,
                            seed=seed,
                            log=True,
                            verbose=True,
                            covar='homogeneous')
    procedure.run()
    
    # Estimate the failure probability
    rho = procedure.rho()
    k = procedure.paramsList[-1].k()
    params = procedure.paramsList[-1]
    print('Done with replicate!')
    
    return rho,k,params

if __name__ == '__main__':
    
    np.random.seed(42)
    
    # Use importance sampling to estimate probability of cascading blackout given
    # a random N-2 contingency.
    
    # x = np.random.normal(10,3,size=(5,dataDim))
    # Hx = h(x)
    
    numReps = 1
    
    # Get random seeds for each replication
    seeds = np.ceil(np.random.uniform(0,99999,size=numReps)).astype(int)
    
    # Define the data used to compute likelihood (i.e. this is the "true" distribution)
    nData = 30
    X = stat.multivariate_normal.rvs(np.zeros((2,)),np.array([[3,0],[0,3]]),size=nData)
    Y = np.sum(X,axis=1,keepdims=True) + stat.norm.rvs(0,.3,size=(nData,1))
    XY = np.concatenate([X,Y],axis=1)
    
    # Define the prior's parameters
    mu_prior = np.zeros(2)
    sigma_prior = np.eye(2)
    sigma = np.eye(2)
    
    # # Create multiprocessing pool w/ 28 nodes for Hyak cluster
    # with mp.Pool(28) as _pool:
    #     result = _pool.map_async(runReplicate,
    #                               list(seeds),
    #                               callback=lambda x : print('Done!'))
    #     result.wait()
    #     resultList = result.get()
    rhoList = []
    paramsList = []
    for seed in list(seeds):
        rho,ce,params = runReplicate(seed,mu_prior,sigma_prior,X)
        rhoList.append(rho)
        paramsList.append(params)
    
    # average all of the GMM densities
    
    
    # make a grid and show the density
    fig,axes = plt.subplots(1,2)
    plotGMM(paramsList, getAvgGmmPDF, axes[0])
    mu_posterior,sigma_posterior = getPosterior(mu_prior, sigma_prior, sigma, X)
    plotMVN(mu_posterior, sigma_posterior, axes[1])
    
    axes[0].set_title('GMM Approximation')
    axes[1].set_title('True Posterior Distribution')
    
    