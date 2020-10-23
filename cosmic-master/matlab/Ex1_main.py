#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 07:29:55 2020

@author: nick
"""

import scipy.stats as stat
import numpy as np
import logging
import datetime as dttm
import matplotlib.pyplot as plt

# Import cem test version (stored locally, NOT a package!)
import cem
from cem import GMMParams

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
    
    bb = 5
    kk = .5
    ee = .1
    
    h_x = (bb-x[:,1]-kk*(x[:,0]-ee)**2) <= 0
    return np.expand_dims(h_x,axis=1)

def r(x):
    
    # Run simulation
    h_x = h(x)
    
    # Compute probability of contingency vector given current parameters
    density = p(x)
    
    return h_x*density

def q(params):
    """
    Get a function for computing the density of the current 
    parametric estimate of the importance sampling distribution.

    Parameters
    ----------
    params : GMMParams
        parameters of the parametric estimate

    Returns
    -------
    q : function
        The density function.

    """
    
    alpha,mu,sigma = params.get()
    
    def _q(X):
        
        if len(X.shape)==2:
            
            x = X
            _alpha = np.tile(alpha,(x.shape[0],1))
            
            # Compute density at each observation
            densities = np.zeros(shape=(x.shape[0],params.k()))
            for j in range(params.k()):
                try:
                    densities[:,j] = stat.multivariate_normal.pdf(x,mu[j,:],sigma[j,:],allow_singular=True)
                except Exception as e:
                    print(sigma[j,:])
                    raise(e)
            
            density = np.expand_dims(np.sum(_alpha*densities,axis=1),axis=1) + jitter
            
            return density
        
        if len(X.shape)==3:
            
            densityList = []
            
            for s in range(X.shape[0]):
                
                x = X[s,:,:]
                _alpha = np.tile(alpha,(x.shape[0],1))
                
                # Compute density at each observation
                densities = np.zeros(shape=(x.shape[0],params.k()))
                for j in range(params.k()):
                    densities[:,j] = stat.multivariate_normal.pdf(x,mu[j,:],sigma[j,:],allow_singular=True)
                
                densityList.append(np.expand_dims(np.sum(_alpha*densities,axis=1),axis=1) + jitter)
            
            return np.concatenate(densityList)
    
    return _q

def generateX(params,num=1):
    """
    Randomly generate a vector from GMM

    Parameters
    ----------
    params: GMMParams
        Estimated importance sampling distribution params.
    num : positive int
        Number of contingencies to generate.

    Returns
    -------
    x : numpy array
        Contingency vector(s).

    """
    # Randomly choose the mixture components to generate from
    alpha,mu,sigma = params.get()
    if params.k() > 1:
        mixtureComponents = np.random.choice(np.array(range(params.k())),size=(num,),p=alpha.squeeze())
    else:
        mixtureComponents = np.zeros(shape=(num,)).astype(int)

    # Generate the vectors from the mixture components
    x = np.zeros(shape=(num,numComponents))
    for i,mixtureComponent in enumerate(mixtureComponents):
        _mu = mu[mixtureComponent,:]
        _sigma = sigma[mixtureComponent,:,:]
        x[i,:] = np.random.RandomState().multivariate_normal(_mu,_sigma)
    
    return x

def expectation(x,q,params):
    """
    Expectation step of EM algorithm.

    Parameters
    ----------
    x : numpy array
        Sampled data from GMM.
    q : function
        The density function for the GMM        
    params : GMMParams
        GMM params.

    Returns
    -------
    gamma : numpy array.
        
    """
    # Tile and reshape arrays for vectorized computation
    alpha,mu,sigma = params.get()
    _alpha = np.tile(alpha,(x.shape[0],1))
    
    # Compute density at each observation
    densities = np.zeros(shape=(x.shape[0],params.k()))
    for j in range(params.k()):
        try:
            densities[:,j] = stat.multivariate_normal.pdf(x,mu[j,:],sigma[j,:],allow_singular=True)
        except Exception as e:
            print(sigma[j,:])
            raise(e)
    #densities += jitter
    
    alpha_q = _alpha*densities
    # Add jitter to density to prevent div by zero error
    density = np.expand_dims(np.sum(alpha_q,axis=1),axis=1) + jitter
    _density = np.tile(density,(1,params.k()))
    
    gamma = alpha_q/_density
    
    return gamma,density

def emIteration(X,r_xList,q,paramsList):
    """
     Use EM algorithm to update parameters.   

    Parameters
    ----------
    X : numpy array
        Sampled data/observations.
    r_xList : list
        List of the r_x numpy array for each sample s
    paramsList : list
        List of the params from all GMM stages


    Returns
    -------
    New GMMParams

    """
    covar_regularization = 10**-6 #Add ridge to covar matrix to prevent numerical instability
    
    gammaList = []
    densityList = []

    for s in range(X.shape[0]):
        
        x = X[s,:,:]
        gamma,density = expectation(x,q,paramsList[s])
        
        gammaList.append(gamma)
        densityList.append(density)

    params = paramsList[-1]
    
    r_div_q_arr = np.zeros(shape=(X.shape[0],1))
    r_div_q_gamma_arr = np.zeros(shape=(X.shape[0],params.k()))
    mu_arr  = np.zeros(shape=(X.shape[0],params.k(),X.shape[2]))
    
    for s in range(X.shape[0]):
        
        # Compute the denominator of alpha_j expression
        r_div_q = np.tile(r_xList[s]/densityList[s],(1,params.k()))
        # Sum over all i, store in array for stage s
        r_div_q_arr[s] = np.sum(r_div_q,axis=0)[0]
        
        # Compute the numerator for alpha_j expression/denominator for mu_j and sigma_j expressions
        r_div_q_gamma = r_div_q*gammaList[s]
        # Sum over all i, store in array for stage s/each j
        r_div_q_gamma_arr[s] = np.sum(r_div_q_gamma,axis=0)
        
        # Tile these quantities so that dimensions match for multiplication
        r_div_q_gamma_tile = np.repeat(np.expand_dims(r_div_q_gamma,axis=-1),X.shape[2],axis=-1)
        x_tile = np.repeat(np.expand_dims(X[s,:,:],axis=1),params.k(),axis=1)
        # Compute numerator of expression for mu_j, sum over all i, store in array for stage s/each j
        mu_arr[s] = np.sum(r_div_q_gamma_tile*x_tile,axis=0)
        
    # Compute new alpha, mu
    alpha = np.sum(r_div_q_gamma_arr,axis=0)/np.sum(r_div_q_arr)
    mu = np.sum(mu_arr,axis=0)/np.repeat(np.expand_dims(np.sum(r_div_q_gamma_arr,axis=0),axis=1),
                                         X.shape[2],axis=1)
    
    sigma_arr = np.zeros(shape=(X.shape[0],params.k(),X.shape[2],X.shape[2]))
    for s in range(X.shape[0]):

        # Tile mu for each sample of stage s
        mu_tile = np.tile(mu,(X.shape[1],1,1))
        # Tile the samples of stage s for each component mean mu_j
        x_tile = np.repeat(np.expand_dims(X[s,:,:],axis=1),params.k(),axis=1)
        # Compute the vector X_i-mu_j
        diff = x_tile-mu_tile
        
        # Compute covar matrices w/ outer product (X_i-mu_j)(X_i-mu_j)^T
        covar = np.zeros(shape=(X.shape[1],params.k(),X.shape[2],X.shape[2]))
        for i in range(X.shape[1]):
            for j in range(params.k()):
                covarMat = np.outer(diff[i,j,:],diff[i,j,:])
                covar[i,j,:,:] = covarMat
        
        # Compute the denominator of expression for sigma_j
        r_div_q = np.tile(r_xList[s]/densityList[s],(1,params.k()))
        r_div_q_gamma = r_div_q*gammaList[s]
        # Tile the denominator to 
        r_div_q_gamma_tile = np.tile(np.expand_dims(r_div_q_gamma,axis=(-1,-2)),(1,1,X.shape[2],X.shape[2]))
    
        sigma_arr[s,:,:,:] = np.sum(r_div_q_gamma_tile*covar,axis=0)
        
    regularization_diag = covar_regularization * np.repeat(np.expand_dims(np.eye(X.shape[2]),
                                                                         axis=0),params.k(),axis=0)
    
    # Compute new sigma
    sigma = np.sum(sigma_arr,axis=0)/\
        np.tile(np.expand_dims(np.sum(r_div_q_gamma_arr,axis=0),axis=(-1,-2)),
                (1,1,X.shape[2],X.shape[2])).squeeze()\
                + regularization_diag #Add ridge to prevent numerical instability
     
    return alpha,mu,sigma

def updateParams(X,rList,q,paramsList,eps=1e-2,maxiter=10,ntrials=10,retCE=False):
    """
    Update params by performing EM iterations until converged

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    rList : TYPE
        DESCRIPTION.
    paramsList: list
        List of all GMM params
    eps : TYPE, optional
        DESCRIPTION. The default is 10**-3.
    maxiter : int
        The maximum number of EM steps to perform.
    ntrials : int
        The number of identical GMM EM algs to run.

    Returns
    -------
    params : TYPE
        DESCRIPTION.

    """
    _params = paramsList[-1]
    paramTrials = [_params.getCopy() for trial in range(ntrials)]
    ceArray = np.zeros(shape=(ntrials,1))
    for trial in range(ntrials):
        
        params = paramTrials[trial]
        i = 0
        delta = np.inf
        
        ce = cem.c_bar(X,rList,q,paramsList,params)
        
        # Loop until EM converges
        while delta >= eps*np.abs(ce) and i < maxiter:
            
            #print('Old Mu: {}'.format(params.get()[1]))
            
            alpha,mu,sigma = emIteration(X, rList, q, paramsList)
            
            #print('New Mu: {}'.format(mu))
            
            params.update(alpha,mu,sigma)
            paramsList[-1] = params
            
            _ce = cem.c_bar(X,rList,q,paramsList,params)
            
            delta = ce-_ce
            #print('Decrease in CE: {}'.format(delta))
            
            ce = _ce
            
            i += 1
        
        # May consider regularization on the log-product of alpha
        ceArray[trial] = ce #- np.log(np.product(alpha)) 
    
    # Choose the best params which decrease cross-entropy the most
    bestParamsInd = np.argmin(ceArray)
    bestParams = paramTrials[bestParamsInd]
    
    # Restore last params in list to be params form stage t-1
    paramsList[-1] = _params
    
    if not retCE:
        return bestParams
    else:
        return bestParams,ceArray[bestParamsInd]

def plotGMM(params,_ax=None,circle=False):
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
        
    contf = ax.contourf(coords,coords,density_grid,levels=10)
    
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
    ax.set_ylim(-5,5)
    
    return ax

def plotStage(s,ax=None):
    
    ax = plotGMM(paramsList[s],ax)

    ax.scatter(X[s,:,0],X[s,:,1],s=2,color='yellow',alpha=.2,zorder=1)
    
def plotStages():
    
    sq = np.ceil(np.sqrt(numIters)).astype(int)
    fig,axes = plt.subplots(sq,sq)
    axes = [ax for sublist in axes for ax in sublist]
    
    for s in range(numIters):
        plotStage(s,axes[s])

if __name__ == '__main__':
    
    timestamp = dttm.datetime.now().strftime('%m%d%Y_%H%M%S')
    logging.basicConfig(filename='experiment_{}.log'.format(timestamp),
                        level=logging.INFO)
    
    np.random.seed(420)
    
    # Use importance sampling to estimate probability of cascading blackout given
    # a random N-2 contingency.
    
    # Mean time to failure for components: start with 10
    numComponents = 2
    # mu = 0 * np.ones(shape=(numComponents,1))
    # Variance of each coordinate in initial GMM
    sigmaSq = 3
    
    #Set a threshold beyond which we consider r_x to be zero to avoid div by zero
    eps = 10**-10
    
    # Set jitter used to prevent numerical issues due to zero densities
    jitter = 10**-100
    
    # Number of different GMM intializations to use
    numGMM = 1
    
    initGMMList = []
    for i in range(numGMM):
        # Initial guess for GMM params
        k = 2
        alpha0 = np.ones(shape=(k,))/k
        
        # Randomly intialize the means of the Gaussian mixture components
        mu0 = np.random.multivariate_normal(np.zeros(numComponents),
                                            np.eye(numComponents),
                                            size=k)
        
        # mu0 = np.random.uniform(-5,5,size=(k,numComponents))
        # mu0 = np.array([[-3,2],
        #                 [3,-1]])
        # Set covariance matrix to be identity
        sigma0 = sigmaSq * np.repeat(np.eye(numComponents)[None,:,:],k,axis=0)
        params = GMMParams(alpha0, mu0, sigma0, numComponents)
        
        initGMMList.append(params)
    
    numIters = 15
    sampleSize = 1000
    paramSize = numComponents
    paramsList = []
    rList = []
    cicArray = np.zeros(shape=(numIters,1))
    
    # Loop for executing the algorithm
    for i in range(numIters):
        
        print('Beginning Stage s={}'.format(i))
        
        # If first stage, need to choose best GMM from init list
        if i==0:
            
            initXList = []
            initRList = []
            updatedInitGMMList = []
            ceArray = np.zeros(shape=(numGMM,1))
            for j,gmmParams in enumerate(initGMMList):
                
                # Generate X from each initGMM
                x = np.random.uniform(-5,5,size=(sampleSize,mu0.shape[1]))
                #x = generateX(paramsList[-1],sampleSize)
            
                # Run simulation and compute likelihood
                try:
                    r_x = r(x)
                except Exception as e:
                    logging.error('Error during r(x): {}'.format(str(e)))
                    raise e
                    
                initRList.append(r_x)
                
                # Update the parameters of the GMM
                _x = np.expand_dims(x,axis=0)
                initXList.append(_x)
                
            
                params,ce = updateParams(initXList[j], [r_x,], q, [gmmParams,], ntrials=10, retCE=True)
                updatedInitGMMList.append(params)
                ceArray[j] = ce
                
            bestParamsInd = np.argmin(ceArray)
            bestParams = updatedInitGMMList[bestParamsInd]
            
            paramsList.append(initGMMList[bestParamsInd])
            paramsList.append(bestParams)
            X = initXList[bestParamsInd]
            rList.append(initRList[bestParamsInd])
            
            
            logging.info('Initial GMM Params:\n'+str(bestParams))
        
        else:
           
            x = generateX(paramsList[-1],sampleSize)
            
            # Run simulation and compute likelihood
            try:
                r_x = r(x)
            except Exception as e:
                logging.error('Error during r(x): {}'.format(str(e)))
                raise e
                
            rList.append(r_x)
            
            # Update the parameters of the GMM
            _x = np.expand_dims(x,axis=0)
    
            X = np.concatenate([X,_x],axis=0)
            
            try:
                params = updateParams(X, rList, q, paramsList, ntrials=10)
            except Exception as e:
                logging.error('Error while updating params: {}'.format(str(e)))
                raise e
                
            # Add new params to the list
            paramsList.append(params)
        
        # Compute the CIC
        try:
            cic = cem.cic(X,rList,q,paramsList)
        except Exception as e:
            logging.error('Error while computing CIC: {}'.format(str(e)))
            raise e
        cicArray[i] = cic
        logging.info('CIC: {}'.format(cic))
