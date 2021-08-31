#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 16:34:57 2021

@author: nick
"""
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt

def _metropolis_hastings(p, d, niter=1000, Sigma=None, x=None):
    
    if Sigma is None:
        Sigma = np.eye(d)
    
    if x is None:
        x = np.zeros((d,))
    
    samples = np.zeros((niter, d))

    # randomly generate the innovation for each step    
    inno = np.random.multivariate_normal(np.zeros((d,)),Sigma,size=niter)
    
    # randomly generate the comparison values
    u = np.random.rand(niter)

    for i in range(niter):
        x_star = x + inno[i]
        px = p(x)
        px_star = p(x_star)

        if u[i] < px_star/px:
            x = x_star
        samples[i] = x

    return samples

def metropolis_hastings(p, d, niter, x=None, Sigma=None):
    
    # do an initial run of 1000 to tune covar matrix of innovation
    samples = _metropolis_hastings(p,d,Sigma=Sigma,x=x)
    Sigma = 2.38**2 * np.cov(samples.T)
    
    # now do the real thing
    samples = _metropolis_hastings(p,d,niter,Sigma,x)
    
    # discard samples from the beginning
    samples = samples[100:]
    
    return samples

def tracePlots(samples):
    
    x = range(samples.shape[0])
    ymin,ymax = np.min(samples),np.max(samples)
    for i in range(samples.shape[1]):
        fig,ax = plt.subplots(1)
        ax.plot(x,samples[:,i])
        
        ax.set_ylim([ymin,ymax])
        ax.set_title('Coordinate {}'.format(i))

# if __name__ == '__main__':
    
    # np.random.seed(123456)
    
    # cov = np.eye(2)
    # mu = np.zeros(2)
    
    # # prior density
    # def p(x):
        
    #     if len(x.shape) > 1:
    #         inPosOrthant = np.all(x>=0,axis=1)
    #     else:
    #         inPosOrthant = np.all(x>=0)
            
    #     density = stat.multivariate_normal.logpdf(x,mean=mu,cov=cov)
        
    #     if len(x.shape) > 1:
    #         density[np.logical_not(inPosOrthant)] = -np.inf
    #     else:
    #         density = density if inPosOrthant else -np.inf

    #     return np.exp(density)
    
    # dataDim = 2
    # nData = 35
    # sigma = 1
    # X = stat.multivariate_normal.rvs(np.zeros((dataDim-1,)),3*np.eye(dataDim-1),size=nData)
    # Y = (np.sum(3*X,axis=1,keepdims=True)+1 if len(X.shape)>1 else 3*X[:,None] + 1) + stat.norm.rvs(0,sigma,size=(nData,1))
    # X = np.concatenate([X,np.ones((X.shape[0],1))],axis=1) if len(X.shape)>1 else np.concatenate([X[:,None],np.ones((X.shape[0],1))],axis=1)
    
    # betaHat = np.linalg.pinv(X.T @ X) @ X.T @ Y
    
    # # likelihood of data under model
    # def l(x):
        
    #     if len(x.shape) > 1:
    #         log_results = np.zeros((x.shape[0],))
    #         for i in range(x.shape[0]):
    #             log_results[i] = np.sum(stat.norm.logpdf(Y-X @ x[i],0,sigma))

    #         log_results = log_results[:,None]
    #     else:
    #         log_results = np.sum(stat.norm.logpdf(Y.squeeze() - X @ x,0,sigma)).astype(np.float128)
        
    #     return np.exp(log_results)
    
    # d = 2
    
    # def g(x):
    #     result = p(x) * l(x)
    #     return result
    
    # samples = metropolis_hastings(g,2,10000,Sigma=2*np.eye(2))
    
    # plt.scatter(samples[:,0],samples[:,1])