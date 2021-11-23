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
import pandas as pd

import CIC.cemSEIS as cem
from CIC.cem import q as getGmmPDF
from CIC.cem import generateX as sampleGmm
from CIC.cem import getAverageDensityFn as getAvgGmmPDF
import CIC.mh as mh

def load_data():
    data = pd.read_csv('customers_data.csv')
    data['Channel'] = data['Channel'] - 1 # 1 is retail, 0 is non-retail channel
    # X = data['Fresh'].values
    # X = data[['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']].values
    X = data[['Fresh','Frozen']].values
    # X = X/np.max(X,axis=0)
    X = X/np.sum(X,axis=1)[:,None]
    Y = data['Channel'].values
    return X,Y[:,None]

def h_mu(theta,mu,sigma):
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
    # beta,logSigma = theta[:,:-1],theta[:,-1]
    beta = theta
    density = stat.multivariate_normal.pdf(beta,mean=mu,cov=sigma)
    # density = np.exp(density_beta + stat.norm.logpdf(logSigma,0,np.sqrt(2)))
    
    if len(beta.shape) > 1:
        density = np.expand_dims(density,axis=1)
    
    return density

def p_x(theta,X,sigma):
    """
    Compute the likelihood of theta given the data X

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
    # beta,logSigma = theta[:,:-1],theta[:,-1]
    beta = theta
    # sigma = np.exp(logSigma)
    log_results = np.zeros((beta.shape[0],))
    _X,_Y = X[:,:-1].astype(np.float128),X[:,-1].astype(np.float128)
    # _X = np.concatenate([_X,np.ones((_X.shape[0],1))],axis=1) if len(X.shape)>1 else np.concatenate([_X[:,None],np.ones((_X.shape[0],1))],axis=1)
    
    if len(beta.shape) > 1:
        for i in range(beta.shape[0]):
            # log_results[i] = np.sum(stat.norm.logpdf(_Y,_X @ beta[i],sigma[i]))
            logit = 1/(1+np.exp(-(_X @ beta[i])))
            log_results[i] = np.sum(np.log(logit)*_Y + np.log(1-logit)*(1-_Y))
        log_results = log_results[:,None]
    
    else:
        logit = 1/(1+np.exp(-(_X @ beta)))
        log_results = np.sum(np.log(logit)*_Y + np.log(1-logit)*(1-_Y))
    
    return log_results

def samplingOracle_mu(n,mu,sigma):
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
    draws = np.zeros((n,mu.size))
    
    beta = stat.multivariate_normal.rvs(mu,sigma,size=n)
    # logSigma = stat.norm.rvs(0,np.sqrt(2),size=n)
    
    # draws[:,:-1] = beta
    # draws[:,-1] = logSigma
    draws = beta
    
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
    
    # # Mark the means of each Gaussian component
    # alpha,mu,sigma = params.get()
    # for j in range(params.k()):
    #     ax.scatter(mu[j,0],mu[j,1],marker='x',color='red')
     
    # if circle:
    #     c1 = plt.Circle((mu[0,0],mu[0,1]),np.linalg.norm(mu)/2,color='red',fill=False)
    #     ax.add_artist(c1)
        
    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_0$')
    ax.set_xlim(-2.5,0)
    ax.set_ylim(-2.5,0)
    
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
        
    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_0$')
    ax.set_xlim(-2.5,0)
    ax.set_ylim(-2.5,0)
    
    return ax

def plotSamples(samples,_ax=None):
    
    if _ax is None:
        fig,ax = plt.subplots(1)
    else:
        ax = _ax
        
    ax.scatter(samples[:,0],samples[:,1])
        
    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_0$')
    ax.set_xlim(-2.5,0)
    ax.set_ylim(-2.5,0)
    
    return ax

def plotSamplesKDE(samples,_ax=None):
    
    if _ax is None:
        fig,ax = plt.subplots(1)
    else:
        ax = _ax
    
    xmin,xmax = -2.5,0
    ymin,ymax = -2.5,0
    k  = stat.gaussian_kde(samples.T)
    Xg, Yg = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([Xg.ravel(), Yg.ravel()])
    Z = np.reshape(k(positions).T, Xg.shape)
    ax.contourf(Xg,Yg,Z,levels=10,cmap='bone')
    
    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_0$')
    ax.set_xlim(-2.5,0)
    ax.set_ylim(-2.5,0)
    
    return ax

# def getPosterior(mu_prior,sigma_prior,sigma,X):
    
#     n = X.shape[0]
#     xBar = np.mean(X,axis=0)

#     sigma_posterior = np.linalg.inv(np.linalg.inv(sigma_prior)+n*np.linalg.inv(sigma))
#     mu_posterior = sigma_posterior@(np.linalg.inv(sigma_prior) @ mu_prior + n*np.linalg.inv(sigma)@xBar)
    
#     return mu_posterior,sigma_posterior

def getPosterior(mu_prior,sigma_prior,sigma,X,Y):
    
    betaHat = np.linalg.pinv(X.T @ X) @ X.T @ Y
    
    Lambda_prior = np.linalg.inv(sigma_prior)
    Lambda_posterior = X.T @ X + Lambda_prior
    sigma_posterior = np.linalg.inv(Lambda_posterior)
    mu_posterior = sigma_posterior @ (X.T @ X @ betaHat + (Lambda_prior @ mu_prior)[:,None])
    
    return mu_posterior.squeeze(),sigma**2 * sigma_posterior

def runReplicate(seed,mu_prior,sigma_prior,XY,sigma):
    
    np.random.seed(seed)
    
    # Run the CEM procedure
    
    def h(theta): 
        return h_mu(theta, mu_prior, sigma_prior)
    
    def p(theta):
        return p_x(theta,XY,sigma)
    
    def samplingOracle(n):
        return samplingOracle_mu(n, mu_prior, sigma_prior)
    
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

    sampleSize = [20000,] + [5000,]*4 + [10000]
    # sampleSize = [5000,] + [2000,]*21 + [3000]
    # sampleSize = [1000,]
    
    procedure = cem.CEMSEIS(initParams,p,samplingOracle,h,
                            numIters=len(sampleSize),
                            sampleSize=sampleSize,
                            seed=seed,
                            log=True,
                            verbose=True,
                            covar='homogeneous',
                            allowSingular=True)
    procedure.run()
    
    # Estimate the failure probability
    rho = procedure.rho()
    k = procedure.paramsList[-1].k()
    params = procedure.paramsList[-1]
    print('Done with replicate!')
    
    return rho,k,params

if __name__ == '__main__':
    
    np.random.seed(123456)
    
    numReps = 10
    
    # Get random seeds for each replication
    seeds = np.ceil(np.random.uniform(0,99999,size=numReps)).astype(int)
    
    # Define the data used to compute likelihood (i.e. this is the "true" distribution)
    nData = 10
    
    # this is the additive noise variance which is known
    sigma = 1
    
    X,Y = load_data()
    X = np.concatenate([X,np.ones((X.shape[0],1))],axis=1) if len(X.shape)>1 else np.concatenate([X[:,None],np.ones((X.shape[0],1))],axis=1)
    XY = np.concatenate([X,Y],axis=1) if len(X.shape)>1 else np.concatenate([X[:,None],Y],axis=1)
    
    dataDim = X.shape[1]
    
    # Define the prior's parameters
    mu_prior = np.zeros(dataDim)
    sigma_prior = np.eye(dataDim)
    
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
        rho,ce,params = runReplicate(seed,mu_prior,sigma_prior,XY,sigma)
        rhoList.append(rho)
        paramsList.append(params)
    
    # fig,axes = plt.subplots(1,2)
    # plotGMM(paramsList, getAvgGmmPDF, axes[0])
    
    # get an approximation using Metropolis-Hastings
    def h(theta): 
        return h_mu(theta, mu_prior, sigma_prior)
    
    def p(theta):
        return p_x(theta,XY,sigma)
    
    def g(x):
        return np.exp(p(x))*h(x)
    
    samples = mh.metropolis_hastings(g,dataDim,50000)
    # mh.tracePlots(samples)
    # plotSamplesKDE(samples, axes[1])
    
    # axes[0].set_title('GMM Approximation')
    # axes[1].set_title('Metropolis-Hastings Approximation')
    
    pdf = getAvgGmmPDF(paramsList)
    
    def getAvgGmmSamples(paramsList,n):
        
        sampleList = []
        
        m = len(paramsList)
        randParamI = np.random.choice(np.arange(0,m),n)
        randDrawCounts = np.bincount(randParamI,minlength=m)
        for i in range(m):
            samples = sampleGmm(paramsList[i],randDrawCounts[i])
            sampleList.append(samples)
        
        samples = np.stack(samples)
        return samples
    
    gmmSamples = getAvgGmmSamples(paramsList, 50000)

    