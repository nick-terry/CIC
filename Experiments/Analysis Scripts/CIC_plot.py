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

import CIC.cemSEIS as cem
from CIC.cem import q as getGmmPDF
from CIC.cem import getAverageDensityFn as getAvgGmmPDF

def h_mu_sigma(theta,mu,sigma):
    """
    Compute the prior density at the parameter values p

    Parameters
    ----------
    theta : numpy array
        The parameter values at which the prior is evaluated
    mu : numpy array
        mean of prior density
    sigma : numpy array
        covar matrix of prior density

    Returns
    -------
    the prior density

    """
        
    density = stat.multivariate_normal.pdf(theta,mean=mu,cov=sigma)
    
    return np.expand_dims(density,axis=1)

def p_x(theta,sigma,X):
    """
    Compute the likelihood of theta given known sigma and the data X

    Parameters
    ----------
    theta : numpy array
        The mean parameter of the data distribution for which the likelihood is being computed.
    sigma : numpy array
        Known covar matrix for data distribution.
    X : numpy array
        Data used to compute likelihood.

    Returns
    -------
    likelihood : numpy array 
        The likelihood of theta.

    """
    
    log_results = np.zeros((theta.shape[0],))
    for i in range(theta.shape[0]):
        log_results[i] = np.sum(stat.multivariate_normal.logpdf(X,theta[i],sigma))
    
    return np.expand_dims(log_results,axis=1)

def samplingOracle_mu_sigma(n,mu,sigma):
    """
    Draw n samples from the prior distribution.

    Parameters
    ----------
    n : integer
        Number of samples to draw.
    mu : numpy array
        mean of prior density
    sigma : numpy array
        covar matrix of prior density

    Returns
    -------
    x : np array
        The drawn samples

    """
    
    x = stat.multivariate_normal.rvs(mu,sigma,size=n)
    
    if len(x.shape)>1:
        return x
    else:
        return np.expand_dims(x,axis=0)


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
    dataDim = 2
    
    np.random.seed(seed)
    
    # Run the CEM procedure
    
    def h(theta): 
        return h_mu_sigma(theta, mu_prior, sigma_prior)
    
    def p(theta):
        return p_x(theta,sigma_prior,X)
    
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

    sampleSize = [2000,] + [500,]*4 + [1000]
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
    
    # make a grid and show the density
    fig,ax = plt.subplots(1)
    coords = np.linspace(-5,5,num=1000)
    coords_grid = np.transpose([np.tile(coords, coords.size),
                                np.repeat(coords, coords.size)])
    
    d = coords_grid.shape[1]
    q_theta = lambda x : stat.multivariate_normal.pdf(x,np.zeros(d),np.eye(d)) * (np.linalg.norm(x,axis=1)**2 < 2)
    density_grid = np.reshape(q_theta(coords_grid),(coords.size,coords.size))
    
    contf = ax.contourf(coords,coords,density_grid,levels=10,cmap='bone')
        
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    