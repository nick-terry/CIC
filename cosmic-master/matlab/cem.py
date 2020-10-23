#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 13:25:44 2020

@author: nick
"""

'''

Implementation of cross-entropy information criterion method from the paper
"Information Criterion for Boltzmann Approximation Problems".

'''
import numpy as np
import copy

class GMMParams:
    
    def __init__(self,alpha,mu,sigma,dataDim):
        
        # Check all the dims
        assert(alpha.shape[0]==mu.shape[0])
        assert(mu.shape[1]==dataDim)
        assert(sigma.shape[0]==alpha.shape[0])
        assert(sigma.shape[1]==dataDim and sigma.shape[2]==dataDim)
        
        self._dataDim = dataDim
        self._k = alpha.shape[0]
        self._alpha = alpha
        self._mu = mu
        self._sigma = sigma
    
    def k(self):
        return self._k
    
    def get(self):
        return self._alpha,self._mu,self._sigma
    
    def update(self,alpha,mu,sigma):
        
        # Check all the dims
        assert(alpha.shape[0]==mu.shape[0])
        assert(mu.shape[1]==self._dataDim)
        assert(sigma.shape[0]==alpha.shape[0])
        assert(sigma.shape[1]==self._dataDim and sigma.shape[2]==self._dataDim)
        
        self._k = alpha.shape[0]
        self._alpha = alpha
        self._mu = mu
        self._sigma = sigma
    
    def getCopy(self):
        return copy.deepcopy(self)
    
    def __str__(self):
        stringRep = ''
        stringRep += 'k={}\n\n'.format(self._k)
        for j in range(self._k):
            stringRep += 'Mixture Component j={}\n'.format(j)
            stringRep += 'alpha={}\n'.format(self._alpha[j])
            stringRep += 'mu={}\n'.format(self._mu[j,:])
            stringRep += 'sigma={}\n\n'.format(self._sigma[j,:,:])
        
        return stringRep

def getVectorizedDensity(densityFn):
    """
    Given a function which computes the density function at an observation
    (1D numpy array), vectorize the function to operate on arrays of observations
    (2D numpy array). The 0-axis is assumed to be indexed by the observation.

    Parameters
    ----------
    densityFn : function
        1D density function.

    Returns
    -------
    vectorized : function
        2D vectorized density function.

    """
    
    vectorized = lambda x : np.apply_along_axis(densityFn, axis=0, arr=x)
    return vectorized

def h(x,r_x,q,params,newParams):
    """
    Compute the entropy of X defined in equation 3.10 of the paper in a
    vectorized manner.

    Parameters
    ----------
    x : numpy array
        The samples draw from the importance sampling distribution. First
        dimension is number of samples, second dimension is the dimension of
        the space from which each observation is drawn (i.e. X \in R^3).
    r_x : numpy array
        The non-negative function r to which the target density is proportional,
        evaluated at the sampled points X. Should have the same shape as X.
    q : function
        Given parameters, returns a density function from
        the posited parametric family.
    params : GMMParams
        Parameters of the previous approximation.
    newParams : GMMParams
        Parameters of the new approximation.

    Returns
    -------
    _h : numpy array
        The entropy of each observation in X w.r.t. the densities given by
        q_eta, q_theta.

    """
    
    q_theta = q(params)
    q_theta_new = q(newParams)
    
    _h = r_x * np.log(q_theta_new(x))/q_theta(x)
    
    return _h

def c_bar(X,rList,q,paramsList,newParams):
    """
    Estimate the cross-entropy from the importance sampling
    distribution defined by eta, using the estimator from equation 3.7 of
    the paper.

    Parameters
    ----------
    X : numpy array
        The samples draw from the importance sampling distribution. First
        dimension is stage, second dimension number of samples, third dimension is the dimension of
        the space from which each observation is drawn (i.e. X \in R^3).
    rList : list
        List containing the non-negative function r to which the target density is proportional,
        evaluated at the sampled points X, for each stage.
    q : function
        Given parameters, returns a density function from
        the posited parametric family.
    paramsList : list
        Parameters of the approximation of the target distribution Q^* at each stage.


    Returns
    -------
    c_hat : float
        The estimated cross-entropy.

    """
    newParams = paramsList[-1]
    
    # Loop over each stage (including the zeroth stage)
    cumulative_c_bar = 0
    for s in range(len(paramsList)):
        x = X[s,:,:]
        r_x = rList[s]
        oldParams = paramsList[s]
        
        cumulative_c_bar += np.sum(h(x,r_x,q,oldParams,newParams))
    
    _c_bar = -1/X.shape[0]/X.shape[1] * cumulative_c_bar
    return _c_bar

def rho(X,rList,q,paramsList):
    """
    Vectorized computation of the consistent estimator of rho given in equation
    3.12 of the paper.

    Parameters
    ----------
    X : numpy array
        The samples draw from the importance sampling distribution. First
        dimension is stage, second dimension number of samples, third dimension is the dimension of
        the space from which each observation is drawn (i.e. X \in R^3).
    rList : list
        List containing the non-negative function r to which the target density is proportional,
        evaluated at the sampled points X, for each stage.
    q : function
        Given parameters, returns a density function from
        the posited parametric family.
    paramsList : list
        Parameters of the approximation of the target distribution Q^* at each stage.

    Returns
    -------
    _rho : float
        The estimate of the normalizing constant rho at the current stage (t-1)

    """
    # Use consistent unbiased estimator for zeroth stage
    if X.shape[0]==1:
        q_theta = q(paramsList[0])
        _rho = np.mean(rList[0]/q_theta(X[0,:,:]))
    
    # Otherwise, use cumulative estimate that excludes zeroth stage
    else:
        cumulativeSum = 0
        
        # Loop over data from each iteration. Exclude initial guess.
        for s in range(1,X.shape[0]):
            q_theta = q(paramsList[s])
            r_x = rList[s]
            cumulativeSum += np.sum(r_x/q_theta(X[s,:,:]))
        
        _rho = cumulativeSum/X.shape[1]/(X.shape[0]-1)
        
    return _rho

def cic(X,rList,q,paramsList):
    """
    Compute the cross-entropy information criterion (CIC) defined in equation 3.13 
    of the paper. This is a more general implementation which does not specify
    the relationship between eta and theta.

    Parameters
    ----------
    X : numpy array
        The samples draw from the importance sampling distribution. First
        dimension is stage, second dimension number of samples, third dimension is the dimension of
        the space from which each observation is drawn (i.e. X \in R^3).
    rList : list
        List containing the non-negative function r to which the target density is proportional,
        evaluated at the sampled points X, for each stage.
    q : function
        Given parameters, returns a density function from
        the posited parametric family.
    paramsList : list
        Parameters of the approximation of the target distribution Q^* at each stage.
        
    Returns
    -------
    _cic : float
        The CIC.

    """

    oldParamsList = paramsList[:-1]
    newParams = paramsList[-1]
    
    # Compute dimension of model parameter space from the number of mixtures, k
    k = newParams.k()
    p = X.shape[2]
    d = (k-1)+k*(p+p*(p+1)/2)
    
    _cic = c_bar(X,rList,q,oldParamsList,newParams) + rho(X,rList,q,paramsList)*d/X.shape[0]/X.shape[1]
    
    return _cic
