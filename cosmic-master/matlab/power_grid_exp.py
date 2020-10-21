#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 07:29:55 2020

@author: nick
"""

import scipy.stats as stat
import scipy.integrate as intgr
import numpy as np
import logging
import datetime as dttm

# Import cem test version (stored locally, NOT a package!)
import cem
import simengine as se

class GMMParams:
    
    def __init__(self,alpha,mu,sigma):
        
        # Check all the dims
        assert(alpha.shape[0]==mu.shape[0])
        assert(mu.shape[1]==numComponents)
        assert(sigma.shape[0]==alpha.shape[0])
        assert(sigma.shape[1]==numComponents and sigma.shape[2]==numComponents)
        
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
        assert(mu.shape[1]==numComponents)
        assert(sigma.shape[0]==alpha.shape[0])
        assert(sigma.shape[1]==numComponents and sigma.shape[2]==numComponents)
        
        self._k = alpha.shape[0]
        self._alpha = alpha
        self._mu = mu
        self._sigma = sigma
        
    def __str__(self):
        stringRep = ''
        stringRep += 'k={}\n\n'.format(self._k)
        for j in range(self._k):
            stringRep += 'Mixture Component j={}\n'.format(j)
            stringRep += 'alpha={}\n'.format(self._alpha[j])
            stringRep += 'mu={}\n'.format(self._mu[j,:])
            stringRep += 'sigma={}\n\n'.format(self._sigma[j,:,:])
        
        return stringRep
        
def prodTerm(t,etaNonFail):
    
    cdf = stat.norm.cdf(t,etaNonFail,sigma)
    return np.product(1-cdf)


def _p(x,eta):
    """
    Compute the probability of having time to failure vector

    Parameters
    ----------
    x : numpy array
        The failure vector.
    eta : numpy array
        The mean time to failure (vector).

    Returns
    -------
    p
        Probability of x.

    """

    nonFails = np.where(x==0)[0]
    fails = np.where(x==1)[0]
        
    integrals = np.zeros(shape=(2,1))
    for i in (0,1):
        
        j = 1 - i
        
        f_j = lambda t : stat.norm.pdf(t,eta[fails[j]],sigma)*\
            prodTerm(t, eta[nonFails])
        innerInt = lambda s : intgr.quad(f_j,s,np.inf)[0]
        
        f_i = lambda s : stat.norm.pdf(s,eta[fails[i]],sigma)*innerInt(s)
        outerInt = intgr.quad(f_i,-np.inf,np.inf)[0]
        
        integrals[i] = outerInt
    
    return np.sum(integrals)

def old_p(x,eta):
    
    if len(x.shape) > 1 and x.shape[0] > 1:
        n = x.shape[0]

        results = np.zeros(shape=(n,1))
        
        for i in range(n):
            results[i] = _p(x[i,:],eta)
            
    else:
        results = _p(x,eta)
        
    return results


def p(x):
    """
    Compute the true likelihood of x from normal distribution

    Parameters
    ----------
    x : numpy array

    Returns
    -------
    likelihood : numpy array

    """
    
    results = np.product(stat.norm.pdf(x,10,3),axis=1)
        
    return np.expand_dims(results,axis=1)

def _runSim(x):
    
    x = np.squeeze(x)
    
    # Pull out the branches that failed and simulate.
    failInd = np.where(x==1)[0]
    br1,br2 = int(failInd[0]),int(failInd[1])
    return sim._simulate(br1,br2)

def runSim(x):
    
    if len(x.shape) > 1 and x.shape[0] > 1:
        n = x.shape[0]

        results = np.zeros(shape=(n,2))
        
        for i in range(n):
            result = _runSim(x[i,:])
            results[i] = result
            
    else:
        results = np.array(_runSim(x)).reshape((1,2))
        
    return results

def r(x):
    
    # Run simulation of contingency
    conting = ttfToContingency(x)
    simResults = runSim(conting)
    
    # Compute probability of contingency vector given current parameters
    density = p(x)
    
    # Need to expand dims here to match density dims
    blackout = np.expand_dims(simResults[:,0],axis=1)
    
    return blackout*density

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
                densities[:,j] = stat.multivariate_normal.pdf(x,mu[j,:],sigma[j,:],allow_singular=True)
            
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

def generateTTF(params,num=1):
    """
    Randomly generate a time to failure vector from current estimate of eta.

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

    # Generate the random time to failures from the mixture components
    timeToFailure = np.zeros(shape=(num,numComponents))
    for i,mixtureComponent in enumerate(mixtureComponents):
        _mu = mu[mixtureComponent,:]
        _sigma = sigma[mixtureComponent,:,:]
        timeToFailure[i,:] = sim.rs.multivariate_normal(_mu,_sigma)
    
    return timeToFailure

def ttfToContingency(timeToFailure):
    # Use failure times to generate contngency vector
    failInd = np.argsort(timeToFailure,axis=1)[:,:2].astype(int)
    x = np.zeros_like(timeToFailure)
    np.put_along_axis(x, failInd, 1, axis=1)
    
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
        densities[:,j] = stat.multivariate_normal.pdf(x,mu[j,:],sigma[j,:],allow_singular=True)
    densities += jitter
    
    alpha_q = _alpha*densities
    # Add jitter to density to prevent div by zero error
    density = np.expand_dims(np.sum(alpha_q,axis=1),axis=1) + jitter
    _density = np.tile(density,(1,params.k()))
    
    gamma = alpha_q/_density
    
    return gamma,density

def emIteration(X,r_xList,q,params):
    """
     Use EM algorithm to update parameters.   

    Parameters
    ----------
    X : numpy array
        Sampled data/observations.
    r_xList : list
        List of the r_x numpy array for each sample s
    params : GMMParams
        The current parameters of the Gaussian Mixture model.

    Returns
    -------
    New GMMParams

    """
    covar_regularization = 10**-6 #Add ridge to covar matrix to prevent numerical instability
    
    gammaList = []
    densityList = []

    for s in range(X.shape[0]):
        
        x = X[s,:,:]
        gamma,density = expectation(x,q,params)
        
        gammaList.append(gamma)
        densityList.append(density)

    r_div_q_arr = np.zeros(shape=(X.shape[0],1))
    r_div_q_gamma_arr = np.zeros(shape=(X.shape[0],params.k()))
    mu_arr  = np.zeros(shape=(X.shape[0],params.k(),X.shape[2]))
    sigma_arr = np.zeros(shape=(X.shape[0],params.k(),X.shape[2],X.shape[2]))
    
    for s in range(X.shape[0]):
        
        r_div_q = np.tile(r_xList[s]/densityList[s],(1,params.k()))
        r_div_q_arr[s] = np.sum(r_div_q,axis=0)[0]
        r_div_q_gamma = r_div_q*gammaList[s]
        
        r_div_q_gamma_arr[s] = np.sum(r_div_q_gamma,axis=0)
        # Tile these quantities so that dimensions match for multiplication
        r_div_q_gamma_tile = np.repeat(np.expand_dims(r_div_q_gamma,axis=-1),X.shape[2],axis=-1)
        x_tile = np.repeat(np.expand_dims(X[s,:,:],axis=1),params.k(),axis=1)
        mu_arr[s] = np.sum(r_div_q_gamma_tile*x_tile,axis=0)
        
    # Compute new alpha, mu
    alpha = np.sum(r_div_q_gamma_arr,axis=0)/np.sum(r_div_q_arr)
    mu = np.sum(mu_arr,axis=0)/np.repeat(np.expand_dims(np.sum(r_div_q_gamma_arr,axis=0),axis=1),
                                         X.shape[2],axis=1)
    
    # Compute new sigma
    for s in range(X.shape[0]):

        mu_tile = np.tile(mu,(X.shape[1],1,1))
        x_tile = np.repeat(np.expand_dims(X[s,:,:],axis=1),params.k(),axis=1)
        diff = x_tile-mu_tile
        
        # Compute covar matrices w/ outer products
        covar = np.zeros(shape=(X.shape[1],params.k(),X.shape[2],X.shape[2]))
        for i in range(X.shape[1]):
            for j in range(params.k()):
                covarMat = np.outer(diff[i,j,:],diff[i,j,:])
                covar[i,j,:,:] = covarMat
        
        r_div_q = np.tile(r_xList[s]/densityList[s],(1,params.k()))
        r_div_q_gamma = r_div_q*gammaList[s]
        r_div_q_gamma_tile = np.tile(np.expand_dims(r_div_q_gamma,axis=(-1,-2)),(1,1,X.shape[2],X.shape[2]))
    
        sigma_arr[s,:,:,:] = np.sum(r_div_q_gamma_tile*covar,axis=0)
        
    regularization_diag = covar_regularization * np.repeat(np.expand_dims(np.eye(X.shape[2]),
                                                                          axis=0),params.k(),axis=0)
    sigma = np.sum(sigma_arr,axis=0)/\
        np.tile(np.expand_dims(np.sum(r_div_q_gamma_arr,axis=0),axis=(-1,-2)),
                (1,1,X.shape[2],X.shape[2])).squeeze()\
                + regularization_diag #Add ridge to prevent numerical instability
     
    return alpha,mu,sigma

def updateParams(X,rList,q,oldParamsList,params,eps=1e-2,maxiter=10):
    """
    Update params by performing EM iterations until converged

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    rList : TYPE
        DESCRIPTION.
    oldParamsList: list
        List of all previous params
    params : TYPE
        DESCRIPTION.
    eps : TYPE, optional
        DESCRIPTION. The default is 10**-3.

    Returns
    -------
    params : TYPE
        DESCRIPTION.

    """
    i = 0
    delta = np.inf
    ce = cem.c_bar(X,rList,q,oldParamsList,params)
    # Loop until EM converges
    while delta >= eps*np.abs(ce) and i < maxiter:
        
        alpha,mu,sigma = emIteration(X, rList, q, params)
        
        params.update(alpha,mu,sigma)
        
        _ce = cem.c_bar(X,rList,q,oldParamsList,params)
        
        delta = np.abs(ce-_ce)
        print(delta)
        
        ce = _ce
        
        i += 1
        
    return params

if __name__ == '__main__':
    
    timestamp = dttm.datetime.now().strftime('%m%d%Y_%H%M%S')
    logging.basicConfig(filename='experiment_{}.log'.format(timestamp),
                        level=logging.INFO)
    
    np.random.seed(42)
    
    # Use importance sampling to estimate probability of cascading blackout given
    # a random N-2 contingency.
    
    # Create a SimulationEngine for evaluating r(x)
    sim = se.SimulationEngine()
    numComponents = sim.numBranches
    
    logging.info('Simulation Case: {}'.format(sim.simulation))
    
    # Mean time to failure for components: start with 10
    mu = 0 * np.ones(shape=(numComponents,1))
    sigma = 3
    
    #Set a threshold beyond which we consider r_x to be zero to avoid div by zero
    eps = 10**-10
    
    # Set jitter used to prevent numerical issues due to zero densities
    jitter = 10**-100
    
    # Initial guess for GMM params
    k = 2
    alpha0 = np.ones(shape=(k,))/k
    # Randomly intialize the means of the Gaussian mixture components
    mu0 = np.random.multivariate_normal(np.zeros(numComponents),
                                        np.ones(shape=(numComponents,numComponents)),
                                        size=k)
    sigma0 = np.ones(shape=(k,numComponents,numComponents))
    params = GMMParams(alpha0, mu0, sigma0)
    
    logging.info('Initial GMM Params:\n'+str(params))
    
    numIters = 2
    sampleSize = 10
    paramSize = numComponents
    paramsList = [params,]
    rList = []
    cicArray = np.zeros(shape=(numIters-1,1))
    
    # Loop for executing the algorithm
    for i in range(numIters-1):
        
        # Randomly generate a time-to-failure vector
        x = generateTTF(params,sampleSize)
        # Create N-2 contingency vector from time-to-failures
        conting = ttfToContingency(x)
        
        # Run simulation and compute likelihood
        try:
            r_x = r(x)
        except Exception as e:
            logging.error('Error during r(x): {}'.format(str(e)))
            raise e
        rList.append(r_x)
        
        # Update the parameters of the GMM
        _x = np.expand_dims(x,axis=0)
        if i==0:
            X = _x
        else:
            X = np.concatenate([X,_x],axis=0)
        
        try:
            params = updateParams(X, rList, q, paramsList, params)
        except Exception as e:
            logging.error('Error while updating params: {}'.format(str(e)))
            raise e
        paramsList.append(params)
        
        # Compute the CIC
        try:
            cic = cem.cic(X,rList,q,paramsList)
        except Exception as e:
            logging.error('Error while computing CIC: {}'.format(str(e)))
            raise e
        cicArray[i] = cic
        logging.info('CIC: {}'.format(cic))
