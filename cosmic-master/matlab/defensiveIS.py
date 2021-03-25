#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 15:42:41 2021

@author: nick
"""

import numpy as np
import scipy.linalg as la

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
    
    # Compute mixture likelihoods
    qx_alpha = np.sum(alpha[None,:] * qx, axis=1, keepdims=True)
    
    # Compute design matrix for regression
    X = np.exp(np.log(qx) - np.log(qx_alpha))
    y = fx * np.exp(np.log(px) - np.log(qx_alpha))
    
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
        The nominal density at each x.
    qx : np array
        The proposal density at each x.
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