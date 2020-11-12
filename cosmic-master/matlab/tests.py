#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 09:29:03 2020

@author: nick
"""

import cem
import matlab
import matlab.engine
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt

'''
Test the Python CEM implementation against the MATLAB implementation using some unit tests
'''

def p(x):
    """
    Compute the true likelihood of x from multivariate normal distribution

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
    
    bb = 5
    kk = .5
    ee = .1
    
    h_x = (bb-x[:,1]-kk*(x[:,0]-ee)**2) <= 0
    return np.expand_dims(h_x,axis=1)

def getEngine():
    eng = matlab.engine.start_matlab()
    return eng

def getCEM(seed):
    
    # Run the CEM procedure
    dataDim = 2
    
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

    procedure = cem.CEM(initParams,p,h,numIters=7,sampleSize=1000,seed=seed,
                        log=False,jitter=0)
    
    return procedure

def npToMAT(array,collapse=True):
    
    if collapse:
        _array = array if len(array.shape)<=2 else np.concatenate(array,axis=0)
        
        arrList = [list(_array[i,:]) for i in range(_array.shape[0])]
        arrM = matlab.double(arrList)
    else:
        _array = array.squeeze()
 
        arrM = matlab.double(_array.tolist())
    
    try:
        assert(np.all((_array-arrM)==0))
    except Exception as e:
        print('Conversion to matlab array failed!')
        raise e
    
    return arrM

def mvgDensityTest(X,proc,eng,epsilon):
    
    # Convert some data to MATLAB format
     
    XM = npToMAT(X)
    
    eng.workspace['XM'] = XM
    eng.workspace['fu'] = eng.eval('@(x) mvnpdf(x,zeros(1,2),eye(2));')
    
    matlabResult = np.array(eng.eval('fu(XM)'))
    pythonResult = proc.p(X)
    
    return np.all(np.abs(matlabResult-pythonResult)<epsilon)

def gmmDensityTest(x,proc,params,eng,epsilon):
    
    pythonDensFn = proc.q(params)
    pythonRes = pythonDensFn(x).squeeze()
    
    # pythonDensFn = proc.log_q(params)
    # pythonRes = np.exp(pythonDensFn(x)).squeeze()
    
    # Convert the params to matlab GMM
    gmm = gmmParamsToMatlab(params, eng)

    xM = npToMAT(x)

    matlabRes = np.array(eng.pdf(gmm,xM)).squeeze()
    
    return np.all(np.abs(matlabRes-pythonRes)<epsilon)

def gmmParamsToMatlab(params,eng,epsilon=1e-10):
    
    # Convert the params to matlab GMM
    alpha,mu,sigma = params.get()
    
    alphaM = matlab.double(list(alpha))
    muM = npToMAT(mu)
    sigmaM = sigma.swapaxes(0,1).swapaxes(1,2)
    sigmaM = npToMAT(sigmaM,False)

    gmm = eng.gmdistribution(muM,sigmaM,alphaM)
    
    eng.workspace['gmm'] = gmm
    muTest = np.array(eng.eval('gmm.mu'))
    sigmaTest = np.array(eng.eval('gmm.Sigma'))
    # If there is more than one GMM component, need to rearrange axes
    if len(sigmaTest.shape)>2:
        sigmaTest = sigmaTest.swapaxes(1,2).swapaxes(0,1)
    # Otherwise, just expand 0 axis
    else:
        sigmaTest = np.expand_dims(sigmaTest,0)
    alphaTest = np.array(eng.eval('gmm.PComponents'))
    
    assert np.all(mu==muTest)
    assert np.all(sigma==sigmaTest)
    assert np.all((alpha-alphaTest)<epsilon)
    
    return gmm

def Hx_WxTest(X,proc,params,eng,epsilon):
    
    gmm = gmmParamsToMatlab(params, eng)
    
    # Convert some data to MATLAB format
    XM = npToMAT(X)
    
    # Compute the other stuff needed  w/ python
    q = proc.q(params)
    # proc.r can only operate on 2D slices
    r = proc.r(X[0,:,:])
    qVal = q(X)
    Hx_Wx = (r/qVal).squeeze()
    
    # Compute Hx_Wx with matlab
    eng.workspace['XM'] = XM
    eng.workspace['fu'] = eng.eval('@(x) mvnpdf(x,zeros(1,2),eye(2));')
    eng.workspace['hw'] = eng.pdf(gmm,XM)
    eng.workspace['Wx'] = eng.eval('fu(XM)./hw')
    Hx_WxM = npToMAT(np.array(eng.workspace['Wx'])*h(X[0,:,:]))
    Hx_WxM = np.array(Hx_WxM).squeeze()
    
    return np.all(np.abs(Hx_Wx-Hx_WxM)<epsilon)

def logpdfTest(X,proc,params,eng,epsilon):
    
    gmm = gmmParamsToMatlab(params, eng)
    eng.workspace['gmm'] = gmm
    # Convert some data to MATLAB format
    
    XM = npToMAT(X)
    
    log_qFn = proc.log_q(params)
    log_q = log_qFn(X)
    log_qM = eng.logdensity(XM,gmm)
    
    return np.all(np.abs(log_q-np.array(log_qM))<epsilon)

def negCETest(X,proc,params,eng,epsilon):
    
    gmm = gmmParamsToMatlab(params, eng)
    
    # Convert some data to MATLAB format
    
    XM = npToMAT(X)
    
    # Compute the other stuff needed  w/ python
    q = proc.q(params)
    # proc.r can only operate on 2D slices
    r = proc.r(X[0,:,:])
    qVal = q(X)
    Hx_Wx = r/qVal
    
    # Compute Hx_Wx with matlab
    eng.workspace['XM'] = XM
    eng.workspace['fu'] = eng.eval('@(x) mvnpdf(x,zeros(1,2),eye(2));')
    eng.workspace['hw'] = eng.pdf(gmm,XM)
    eng.workspace['Wx'] = eng.eval('fu(XM)./hw')
    Hx_WxM = npToMAT(np.array(eng.workspace['Wx'])*h(X[0,:,:]))
    
    matlabRes = eng.negCE(XM,Hx_WxM,gmm)
    pythonRes = -proc.c_bar(X,[r,],proc.q,proc.log_q,[params,],params)
    
    return np.abs(matlabRes-pythonRes)<epsilon

def rhoTest(X,proc,params,eng,epsilon):
    
    gmm = gmmParamsToMatlab(params, eng)
    
    # Convert some data to MATLAB format
    
    XM = npToMAT(X)
    
    # Compute the other stuff needed  w/ python
    q = proc.q(params)
    # proc.r can only operate on 2D slices
    r = proc.r(X[0,:,:])
    qVal = q(X)
    Hx_Wx = r/qVal
    
    # Compute Hx_Wx with matlab
    eng.workspace['XM'] = XM
    eng.workspace['fu'] = eng.eval('@(x) mvnpdf(x,zeros(1,2),eye(2));')
    eng.workspace['hw'] = eng.pdf(gmm,XM)
    eng.workspace['Wx'] = eng.eval('fu(XM)./hw')
    eng.workspace['Hx_Wx'] = npToMAT(np.array(eng.workspace['Wx'])*h(X[0,:,:]))
    
    matlabRes = eng.eval('mean(Hx_Wx)')
    pythonRes = proc.rho(proc.X,proc.rList,proc.q,proc.paramsList)
    
    return np.abs(matlabRes-pythonRes)<epsilon

if __name__=='__main__':
    
    epsilon = 1e-10
    
    seed = 42
    eng = getEngine()
    eng.cd('/Users/nick/GitHub/CIC-Paper/CIC_example_Matlab_code')
    proc = getCEM(seed)
    
    X = proc.generateX(proc.initParams,num=1000)
    X = X.reshape((1,X.shape[0],X.shape[1]))
    
    # Check that MVG/GMM density is computed correctly
    assert mvgDensityTest(X, proc, eng, epsilon)
    assert gmmDensityTest(X, proc, proc.initParams, eng, epsilon)
    
    # Check that log-density of GMM is computed correctly
    assert logpdfTest(X, proc, proc.initParams, eng, epsilon)
    
    # Check that Hx_Wx is computed correctly
    assert Hx_WxTest(X, proc, proc.initParams, eng, epsilon)
    
    # Check that cross-entropy is computed correctly
    assert negCETest(X, proc, proc.initParams, eng, epsilon)
    
    # Run a stage so we can check rho computations
    bestParamsByK,cicByK,cicMA = proc.runStage(0,proc.initParams,kMin=1)
    
    # Select the best params from the stage by minimizing CIC
    bestInd = np.argmin(cicByK)
    bestParams = bestParamsByK[bestInd]
    
    # Add new params to the list
    proc.paramsList.append(bestParams)
    proc.cicArray[0] = cicByK[bestInd]
    
    # Check that rho is computed correctly
    assert rhoTest(proc.X,proc,proc.paramsList[-1],eng,epsilon)
    