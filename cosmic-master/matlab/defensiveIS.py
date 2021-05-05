#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 15:42:41 2021

@author: nick
"""

import numpy as np
import scipy.linalg as la
import warnings
from scipy.special import logsumexp

'''
Implementing some methods from the paper Safe and Effective Importance Sampling
by Owen and Zhou.
'''

def getBeta(fx,px,qx,alpha):
    """
    Computes least squares estimates of the control variates coefficients beta.

    Parameters
    ----------
    fx : np array
        Function being integrated at each x.
    px : np array
        The nominal density at each x
    qx : np array
        The likelihood of each x for each mixture of the proposal density.
    alpha : np array
        The weights of each mixture component for the proposal density

    Returns
    -------
    beta : np array
        The least squares coefficients.

    """
    
    # Check that we have not mixture components equal to zero
    # try:
    #     assert(not np.any(alpha==0))
    # except Exception as e:
    #     print('Bad alpha in SEIS!')
    #     print(alpha)
    #     raise e
    
    # Compute design matrix for regression
    
    # Compute where we have at least one component with non-zero likelihood
    # nzi = np.sum(qx>0,axis=1,keepdims=True)
    nzi = np.bitwise_or.reduce(qx>0,axis=1)
    
    # Compute mixture likelihoods
    with np.errstate(divide='ignore'):
        tmp = np.log(alpha[None,:]) + qx
    
    tmp[np.isneginf(tmp)] = 0
    # qx_alpha = np.sum(tmp, axis=1, keepdims=True)
    qx_alpha = logsumexp(tmp,axis=1,keepdims=True)
    
    # Check that qx_alpha computation didn't produce any NaN/inf values
    # try:
    #     assert(not np.any(np.isnan(qx_alpha)))
    #     assert(not np.any(np.isinf(qx_alpha)))
    # except Exception as e:
    #     print('Bad qx_alpha in SEIS!')
    #     print(qx_alpha)
    #     raise e
    
    # try:
    #     assert(not np.any(np.bitwise_and.reduce(qx[nzi]==0,axis=1)))
    # except Exception as e:
    #     print('Non-zero indices not working for qx!')
    #     raise e
    
    # try:
    #     assert(not np.any(qx_alpha[nzi]==0))
    # except Exception as e:
    #     print('Non-zero indices not working for qx_alpha!')
    #     print(alpha)
    #     raise e
    
    # X = np.zeros_like(qx)
    
    # Need to catch warnings here because we may take a log of zero
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     X[nzi] = np.exp(np.log(qx[nzi]) - np.log(qx_alpha[nzi]))
    # X[X==-np.inf] = 0
    X = np.exp(qx - qx_alpha)
    
    # y = fx * np.exp(np.log(px) - np.log(qx_alpha))
    y = fx * np.exp(px - qx_alpha)
    
    # X = np.exp(np.log(qx) - np.log(qx_alpha))
    # y = fx * np.exp(np.log(px) - np.log(qx_alpha))
    
    # Perform SVD on X to compute regression coefficients
    # U,S,V = np.linalg.svd(X)
    
    # beta = V @ np.diag(1/S) @ U.T @ y
    
    # Compute beta using pseudo-inverse of X^T X
    Xpinv = la.pinvh(X.T @ X)
    beta = Xpinv @ (X.T @ y)
    
    return beta,qx,qx_alpha

def getBeta2Mix(fx,px,qx,alpha):
    """
    Computes least squares estimates of the control variates coefficients beta
    in the special case where the proposal mixture only consists of the nominal
    density and a single proposal density.

    Parameters
    ----------
    fx : np array
        Function being integrated at each x.
    px : np array
        log of The nominal density at each x.
    qx : np array
        log of The proposal density at each x.
    alpha : float
        The weight of the nominal density in the mixture.

    Returns
    -------
    beta : np array
        The least squares coefficients.

    """
    
    _alpha = np.array([alpha, 1-alpha])
    _qx = np.zeros((qx.shape[0],2))
    _qx[:,0] = px.squeeze()
    _qx[:,1] = qx.squeeze()

    beta,_qx,qx_alpha = getBeta(fx, px, _qx, _alpha)
    
    return beta,_qx,qx_alpha