#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 18:19:04 2021

@author: nick
"""
import numpy as np
import logging
from scipy.special import logsumexp
import warnings

import CIC.cem as cem
import CIC.defensiveIS as defensiveIS
import copy

def splitExpSumLog(x,y=None):
    """
    Compute the product of x and exp(y) using log transforms. x is not assumed
    to be positive

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if y is None:
        y = np.zeros_like(x)
        
    posPart = np.ma.log(np.ma.masked_array(x,x<=0)) + y
    negPart = np.ma.log(np.ma.masked_array(-x,x>=0)) + y
    
    with warnings.catch_warnings():

        warnings.filterwarnings('error')
        
        try:
            product = np.exp(posPart).filled(0) - np.exp(negPart).filled(0)
            
        except Warning as w:
            # print('Positive Part: {}'.format(posPart))
            # print('Negative Part: {}'.format(negPart))
            print(w)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                posExp,negExp = np.exp(posPart).filled(0),np.exp(negPart).filled(0)
                product = posExp - negExp
        
    # product = np.exp(posPart).filled(0) - np.exp(negPart).filled(0)
        
    return product,posPart,negPart

class GMMParams:
    
    def __init__(self,alpha,mu,sigma,dataDim):
        
        # Check all the dims
        assert(alpha.shape[0]==mu.shape[0])
        assert(mu.shape[1]==dataDim)
        assert(sigma.shape[0]==alpha.shape[0])
        assert(sigma.shape[1]==dataDim and sigma.shape[2]==dataDim)
        
        assert(not np.any(np.isnan(alpha)))
        assert(not np.any(np.isnan(mu)))
        assert(not np.any(np.isnan(sigma)))
        
        # Make sure we don't have a nearly-singular covar matrix
        for i in range(alpha.shape[0]):
            assert(np.any(np.abs(sigma[i])>1e-100))
            
        self._dataDim = dataDim
        self._k = alpha.shape[0]
        self._alpha = alpha
        self._mu = mu
        self._sigma = sigma
    
    def k(self):
        return self._k
    
    def dim(self):
        return self._dataDim
    
    def get(self):
        return self._alpha,self._mu,self._sigma
    
    def update(self,alpha,mu,sigma):
        
        # Check all the dims
        assert(alpha.shape[0]==mu.shape[0])
        assert(mu.shape[1]==self._dataDim)
        assert(sigma.shape[0]==alpha.shape[0])
        assert(sigma.shape[1]==self._dataDim and sigma.shape[2]==self._dataDim)
        
        assert(not np.any(np.isnan(alpha)))
        assert(not np.any(np.isnan(mu)))
        assert(not np.any(np.isnan(sigma)))
        
        
        # Make sure we don't have a nearly-singular covar matrix
        for i in range(alpha.shape[0]):
            assert(np.any(np.abs(sigma[i])>1e-100))
        
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

class CEMSEIS(cem.CEM):
    
    def __init__(self,initParams,p,samplingOracle,h,alpha=.1,gamma=None,**kwargs):
        """
        Create a new CEMSEIS 
        (Cross-Entropy Minimizer w/ Safe and Effective Importance Sampling).

        Parameters
        ----------
        initParams : GMMParams
            The initial parameters of the GMM model.
        p : function
            Returns the true density (likelihood) at a given point.
        samplingOracle : function
            Returns samples from the true/nominal density
        h : function
            Simulator evaluation
        alpha : float in (0,1),optional
            The weight given to the nominal density in the proposal mixture.
            Default is .1 as recommended in Owen and Zhou.
        **kwargs : dict
            Keyword arguments.

        """
        
        self.initParams = initParams
        self.paramsList = [initParams,]
        self.dim = initParams.get()[1].shape[1]
        self.p = p
        self.samplingOracle = samplingOracle
        self.h = h
        self.alpha = alpha 
        self.gamma = gamma if gamma is not None else 100
        
        # Store some likelihoods/function evals to avoid recomputing
        self.hList = []
        self.pxList = []
        self.qxList = []
        self.logpxList = []
        self.logqxList = []
        
        self.rList = []
        self.Hx_WxList = []

        # Store some historical data as procedure is executed
        self.bestParamsLists = []
        self.cicLists = []
        
        self.errored = False
        
        # Some important constants which can be changed     
        # Set jitter used to prevent numerical issues due to zero densities
        if 'jitter' in kwargs:
            self.jitter = kwargs['jitter']
        else:
            self.jitter = 0
          
        if 'numIters' in kwargs:
            self.numIters = kwargs['numIters']
        else:
            self.numIters = 10
        
        if 'sampleSize' in kwargs:
            self.sampleSize = kwargs['sampleSize']
            
            # Check if an iterable was passed
            try:
                iter(self.sampleSize)
                assert(len(self.sampleSize)==self.numIters)
                self.variableSampleSizes = True
                
            except AssertionError as e:
                print('Incorrect number of variable sample sizes provided!')
                raise e
                
            except:
                self.variableSampleSizes = False
        else:
            self.sampleSize = 1000
            self.variableSampleSizes = False
            
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])
            
        if 'allowSingular'in kwargs:
            self.allowSingular = kwargs['allowSingular']
        else:
            self.allowSingular = False
            
        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
            assert(type(self.verbose)==bool)
        else:
            self.verbose = False
            
        if 'log' in kwargs:
            self.log = kwargs['log']
        else:
            self.log = True
            
        if 'covar' in kwargs:
            self.covarStruct = kwargs['covar']
        else:
            self.covarStruct = 'full'
            
        if 'repNum' in kwargs:
            self.repNum = kwargs['repNum']
        else:
            self.repNum = -1
            
        if 'seis' in kwargs:
            self.seis = kwargs['seis']
        else:
            self.seis = True
            
        self.cicArray = np.zeros(shape=(self.numIters,1))
        
        # self.covar_regularization = 10**-100
        self.covar_regularization = 0
    
    def c_bar(self,params,gmm=False):
        """
        Estimate the cross-entropy from the importance sampling
        distribution defined by eta, using the estimator from equation 3.7 of
        the paper.
        
        See eqn 3.18
    
        Parameters
        ----------
        params : list
            Parameters of the approximation we wish to estimate cross entropy for
    
    
        Returns
        -------
        _c_bar : float
            The estimated cross-entropy.
    
        """
        if not gmm:
            X = np.concatenate(self.X,axis=0)
        else:
            X = np.concatenate(self.X_gmm,axis=0)
            
        log_q_theta = self.log_q(params)
        llh = log_q_theta(X)
        
        n = llh.shape[0]
        
        # Compute c_bar w/o log transform   
        Hx = np.concatenate(self.hList, axis=0)

        px = np.concatenate(self.logpxList, axis=0)
        qx = np.concatenate(self.logqxList, axis=0)
        
        qx_mix = np.exp(np.log(self.alpha) + px) + np.exp(np.log(1-self.alpha) + qx)
        Wx = px/qx_mix
        
        c_bar = -np.sum(Hx * Wx * llh)/n
            
        return c_bar
    
    def rho(self):
        """
        Vectorized computation of estimator of event probability. Using the MCV
        estimator from "Safe and Effective Importance Sampling", Owen and Zhou.
        (Equation 16)
    
        Returns
        -------
        _rho : float
            The estimate of the normalizing constant rho at the current stage (t-1)
    
        """
        
        # Catch warnings here bc overflow may happen
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Use consistent unbiased estimator for zeroth stage
            if len(self.X)==1:
                
                # Compute the control coefficients beta
                fx = np.concatenate(self.hList, axis=0)
                # px = np.concatenate(self.pxList, axis=0)
                # qx = np.concatenate(self.qxList, axis=0)
                px = np.concatenate(self.logpxList, axis=0)
                qx = np.concatenate(self.logqxList, axis=0)
            
            # Otherwise, use cumulative estimate that excludes zeroth stage
            else:
                
                # Compute the control coefficients beta
                fx = np.concatenate(self.hList[1:], axis=0)
                # px = np.concatenate(self.pxList[1:], axis=0)
                # qx = np.concatenate(self.qxList[1:], axis=0)
                px = np.concatenate(self.logpxList[1:], axis=0)
                qx = np.concatenate(self.logqxList[1:], axis=0)
    
            if not self.seis:
                _rho = np.mean(fx * np.exp(px-qx))
            
            else:
                beta,_qx,qx_alpha = defensiveIS.getBeta2Mix(fx,px,qx,self.alpha)
                _qx,qx_alpha = _qx.astype(np.float128),qx_alpha.astype(np.float128)
                    
                _rho = np.mean((np.exp(np.log(fx) + px) - np.exp(_qx) @ beta)/np.exp(qx_alpha)) + np.sum(beta)
                
                numerator = fx * np.exp(px) - np.exp(_qx) @ beta
                product,posPart,negPart = splitExpSumLog(numerator)
                
                product,posPart,negPart = splitExpSumLog(np.repeat(beta,_qx.shape[0],axis=1).T, _qx)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # _rho = np.mean((fx * np.exp(px) -\
                    #                 (np.exp(posPart-).filled(0)-np.exp(negPart).filled(0)))/qx_alpha) + np.sum(beta)
                        
                    _rho = np.mean(fx * np.exp(px-qx_alpha) -\
                                    np.sum(np.exp(posPart-qx_alpha).filled(0)-\
                                    np.exp(negPart-qx_alpha).filled(0),axis=-1,keepdims=True)) + np.sum(beta)
            
        try:
            assert(not np.isnan(_rho))
            assert(_rho>=0)
        except Exception as e:
            raise e
            
        return _rho
    
    def generateX(self,params,num=1):
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
        if params is None:
            x = self.samplingOracle(num)
            
        else:
            # Randomly choose the mixture components to generate from
            alpha,mu,sigma = params.get()
            if params.k() > 1:
                try:
                    mixtureComponents = np.random.choice(np.array(range(params.k())),size=(num,),p=alpha.squeeze().astype(np.float64))
                except Exception as e:
                    print(alpha.squeeze())
                    print(np.sum(alpha.squeeze()))
                    raise e
            else:
                mixtureComponents = np.zeros(shape=(num,)).astype(int)
        
            # Generate the vectors from the mixture components
            x = np.zeros(shape=(num,self.dim))
            for i,mixtureComponent in enumerate(mixtureComponents):
                _mu = mu[mixtureComponent,:]
                _sigma = sigma[mixtureComponent,:,:]
                x[i,:] = np.random.RandomState().multivariate_normal(_mu,_sigma)
        
        return x.astype(np.longdouble)
    
    def emIteration(self,params):
        """
         Use EM algorithm to update parameters.   
    
        Parameters
        ----------
        params : GMMParams
            The params to update with EM.
    
    
        Returns
        -------
        alpha : numpy array
        mu : numpy array
        sigma: numpy array
    
        """
        covar_regularization = self.covar_regularization #Add ridge to covar matrix to prevent numerical instability
        
        # To make this code (slightly) more readable

        # X,q = np.concatenate(self.X_gmm,axis=0),self.q
        X,q = np.concatenate(self.X,axis=0),self.q
        
        # gamma = self.expectation(X, q, params)
        # Added small amount of regularization here to prevent a component having alpha=0
        try:
            log_gamma =  self.log_expectation(X, q, params) #+ self.covar_regularization
        except:
            return -1,-1,-1
        
        log_r_div_q = np.concatenate(self.Hx_WxList,axis=0)
        
        # We need to avoid taking log of zero entries here
        # r_div_q_gamma = np.zeros_like(log_gamma)
        # nzi = np.squeeze(r_div_q>0) 
        
        # try:
        #     assert(np.all(gamma[nzi]>0))
        # except Exception as e:
        #     print('Some gamma values are negative!')
        #     raise e
        
        # This works since multiplying by the zero entries still yields zero
        # r_div_q_gamma[nzi] = np.exp( np.log(r_div_q[nzi]) + log_gamma[nzi] )
        log_r_div_q_gamma = log_r_div_q + log_gamma
        # r_div_q_gamma = np.exp(log_r_div_q_gamma)
        
        # Compute new alpha, mu
        log_sum_r_div_q_gamma = logsumexp(log_r_div_q_gamma,axis=0,keepdims=True)
        log_sum_r_div_q = logsumexp(log_r_div_q)
        
        # TODO: this assertion is disabled as it may no longer be true that -inf entries break the EM algo
        if np.any(log_sum_r_div_q_gamma==-np.inf):
            # print('zeros in r_div_q_gamma!')
            # print('k={}'.format(params.k()))
            # print(_sum_r_div_q_gamma)
            # raise Exception
            
            # If this happens, return a -1 for alpha to indicate an error
            return -1,None,None
            
        alpha = np.exp(log_sum_r_div_q_gamma.squeeze(0) - log_sum_r_div_q)
        
        # Check that the mixing proportions sum to 1
        try:
            assert(np.abs(np.sum(alpha)-1)<1e-3)
        except Exception as e:
            if self.log:
                logging.warning('alpha computed during M step does not sum to one! Normalizing alpha...')
            alpha = alpha/np.sum(alpha)
        
        # Check that no mixture components have zero weight
        if np.any(alpha==0):
            # If this happens, return a -1 for alpha to indicate an error
            return -1,None,None
        
        # Check that there is no inf/nan values
        try:
            assert(not np.any(np.isnan(alpha)))
            assert(not np.any(alpha==np.inf))
        except Exception as e:
            print('Floating point error while computing alpha!')
            raise e

        # mu = np.sum(r_div_q_gamma[:,:,None]*X[:,None,:],axis=0)/np.sum(r_div_q_gamma,axis=0,keepdims=True).T
        # r_div_q_gamma_normed = np.exp(log_r_div_q_gamma[:,:,None]-np.repeat(log_sum_r_div_q_gamma.T,X.shape[-1],axis=1))
        r_div_q_gamma_normed = np.exp(log_r_div_q_gamma - log_sum_r_div_q_gamma)
        # product,posPart,negPart = splitExpSumLog(X[:,None,:],log_r_div_q_gamma[:,:,None])
        # product,posPart,negPart = splitExpSumLog(X[:,None,:],r_div_q_gamma_normed)
        # try:
        #     mu = np.sum(np.exp(posPart - np.repeat(log_sum_r_div_q_gamma.T,posPart.shape[-1],axis=1)).filled(0) -\
        #                 np.exp(negPart - np.repeat(log_sum_r_div_q_gamma.T,posPart.shape[-1],axis=1)).filled(0),axis=0)
        #     mu = np.sum(np.exp(posPart).filled(0) -\
        #                 np.exp(negPart).filled(0),axis=0)
        # except Exception as e:
        #     raise e
        
        mu = np.sum(r_div_q_gamma_normed[:,:,None]*X[:,None,:],axis=0)
        
        if len(mu.shape)==1:
            mu = mu[None,:]
        
        # Check that there is no inf/nan values
        try:
            assert(not np.any(np.isnan(mu)))
            assert(not np.any(mu==np.inf))
        except Exception as e:
            print('Floating point error while computing mu!')
            raise e
        
        # If using full covar structure, compute the covar matrix using outer productfor full sample covar
        if self.covarStruct == 'full':
            if mu.shape[0]==1:
                # Compute difference of data from mean vector
                diff = X-mu
                # Compute sigma using a vectorized outer product. 
                # See https://stackoverflow.com/questions/42378936/numpy-elementwise-outer-product
                
                # sigma = np.sum(r_div_q_gamma[:,:,None]*(diff[:,:,None]*diff[:,None]),axis=0)/\
                #     np.sum(r_div_q_gamma,axis=0) + covar_regularization * np.eye(mu.shape[-1])
                
                # product,posPart,negPart = splitExpSumLog((diff[:,:,None]*diff[:,None]),log_r_div_q_gamma[:,:,None])
                # try:
                #     sigma = np.sum(np.exp(posPart - np.repeat(log_sum_r_div_q_gamma.T,posPart.shape[-1],axis=1)).filled(0) -\
                #                 np.exp(negPart - np.repeat(log_sum_r_div_q_gamma.T,posPart.shape[-1],axis=1)).filled(0),axis=0) +\
                #                 covar_regularization * np.eye(mu.shape[-1])
                # except Exception as e:
                #     raise e
                
                sigma = np.sum(r_div_q_gamma_normed[:,:,None]*(diff[:,:,None]*diff[:,None]),axis=0)\
                    + covar_regularization * np.eye(mu.shape[-1])
                
                if len(sigma.shape)==2:
                    sigma = sigma[None,:,:]
            
            else:
                # Compute difference of data from mean vector
                diff = X[:,None,:]-mu
                # Compute sigma using a vectorized outer product. 
                # See https://stackoverflow.com/questions/42378936/numpy-elementwise-outer-product
                # sigma = np.sum(r_div_q_gamma[:,:,None,None]*(diff[:,:,:,None]*diff[:,:,None]),axis=0)/\
                #     np.sum(r_div_q_gamma,axis=0,keepdims=True).T[:,:,None] + covar_regularization * np.eye(mu.shape[-1])
                    
                # product,posPart,negPart = splitExpSumLog((diff[:,:,:,None]*diff[:,:,None]),log_r_div_q_gamma[:,:,:,None])
                # try:
                #     sigma = np.sum(np.exp(posPart - np.repeat(log_sum_r_div_q_gamma.T,posPart.shape[-1],axis=1)).filled(0) -\
                #                 np.exp(negPart - np.repeat(log_sum_r_div_q_gamma.T,posPart.shape[-1],axis=1)).filled(0),axis=0) +\
                #                 covar_regularization * np.eye(mu.shape[-1])
                # except Exception as e:
                #     raise e
                    
                sigma = np.sum(r_div_q_gamma_normed[:,:,None,None]*(diff[:,:,:,None]*diff[:,:,None]),axis=0)\
                     + covar_regularization * np.eye(mu.shape[-1])
                
                if len(sigma.shape)==2:
                    sigma = sigma[None,:,:]
                
        # If using homogeneous covariance structure, covar matrix is a scalar times identity matrix.
        # We compute the scaling factor using an EM update equation
        elif self.covarStruct == 'homogeneous':
            if mu.shape[0]==1:
                # Compute difference of data from mean vector
                diff = X-mu
                # Compute sigma using a vectorized outer product. 
                # See https://stackoverflow.com/questions/42378936/numpy-elementwise-outer-product
                # sigma = np.sum(r_div_q_gamma*np.sum(diff**2,axis=-1,keepdims=True),axis=0)[:,None]/\
                #     np.sum(r_div_q_gamma,axis=0)/X.shape[-1]
                # sigma = (sigma + covar_regularization) * np.eye(X.shape[-1])
                
                # product,posPart,negPart = splitExpSumLog(np.sum(diff**2,axis=-1,keepdims=True),log_r_div_q_gamma)
                # try:
                #     sigma = np.sum(np.exp(posPart - np.repeat(log_sum_r_div_q_gamma.T,posPart.shape[-1],axis=1)).filled(0) -\
                #                 np.exp(negPart - np.repeat(log_sum_r_div_q_gamma.T,posPart.shape[-1],axis=1)).filled(0),axis=0)
                #     sigma = (sigma + covar_regularization) * np.eye(X.shape[-1])
                # except Exception as e:
                #     raise e
                
                sigma = np.sum(r_div_q_gamma_normed*np.sum(diff**2,axis=-1,keepdims=True),axis=0)[:,None]/X.shape[-1]
                sigma = (sigma + covar_regularization) * np.eye(X.shape[-1])
                
                if len(sigma.shape)==2:
                    sigma = sigma[None,:,:]
        
            else:
                # Compute difference of data from mean vector
                diff = X[:,None,:]-mu
                # Compute sigma using a vectorized outer product. 
                # See https://stackoverflow.com/questions/42378936/numpy-elementwise-outer-product
                # sigma = np.sum(r_div_q_gamma[:,:,None]*np.sum(diff**2,axis=-1,keepdims=True),axis=0)/\
                #     np.sum(r_div_q_gamma,axis=0,keepdims=True).T/X.shape[-1] + covar_regularization
                # sigma = np.repeat(np.repeat(sigma[:,:,None],axis=1,repeats=X.shape[-1]),
                #                   axis=2,repeats=X.shape[-1]) * np.eye(X.shape[-1])
                
                # product,posPart,negPart = splitExpSumLog(np.sum(diff**2,axis=-1,keepdims=True),log_r_div_q_gamma[:,:,None])
                # try:
                #     sigma = np.sum(np.exp(posPart - np.repeat(log_sum_r_div_q_gamma.T,posPart.shape[-1],axis=1)).filled(0) -\
                #                 np.exp(negPart - np.repeat(log_sum_r_div_q_gamma.T,posPart.shape[-1],axis=1)).filled(0),axis=0) +\
                #                 covar_regularization
                #     sigma = np.repeat(np.repeat(sigma[:,:,None],axis=1,repeats=X.shape[-1]),
                #                 axis=2,repeats=X.shape[-1]) * np.eye(X.shape[-1])
                     
                # except Exception as e:
                #     raise e
                
                sigma = np.sum(r_div_q_gamma_normed[:,:,None]*np.sum(diff**2,axis=-1,keepdims=True),axis=0)/X.shape[-1] + covar_regularization
                sigma = np.repeat(np.repeat(sigma[:,:,None],axis=1,repeats=X.shape[-1]),
                                  axis=2,repeats=X.shape[-1]) * np.eye(X.shape[-1])
                
                # print([np.all(sigma[i]==0) for i in range(sigma.shape[0])])
                # print(np.any([np.all(sigma[i]==0) for i in range(sigma.shape[0])]))

                if len(sigma.shape)==2:
                    sigma = sigma[None,:,:]
                    
            
        elif self.covarStruct=='diagonal':
            if mu.shape[0]==1:
                # Compute difference of data from mean vector
                diff = X-mu
                # Compute sigma using a vectorized outer product. 
                # See https://stackoverflow.com/questions/42378936/numpy-elementwise-outer-product
                # sigma = np.sum(r_div_q_gamma * diff**2,axis=0)/\
                #     np.sum(r_div_q_gamma,axis=0)/X.shape[-1]
                # sigma = np.diag(sigma) + covar_regularization * np.eye(X.shape[-1])
                
                # product,posPart,negPart = splitExpSumLog(diff**2,log_r_div_q_gamma)
                # try:
                #     sigma = np.sum(np.exp(posPart - np.repeat(log_sum_r_div_q_gamma.T,posPart.shape[-1],axis=1)).filled(0) -\
                #                 np.exp(negPart - np.repeat(log_sum_r_div_q_gamma.T,posPart.shape[-1],axis=1)).filled(0),axis=0)
                #     sigma = np.diag(sigma)/params.dim() + covar_regularization * np.eye(X.shape[-1])
                     
                # except Exception as e:
                #     raise e
                
                sigma = np.sum(r_div_q_gamma_normed * diff**2,axis=0)/X.shape[-1]
                sigma = np.diag(sigma) + covar_regularization * np.eye(X.shape[-1])
                
                if len(sigma.shape)==2:
                    sigma = sigma[None,:,:]
        
            else:
                # Compute difference of data from mean vector
                diff = X[:,None,:]-mu
                # Compute sigma using a vectorized outer product. 
                # See https://stackoverflow.com/questions/42378936/numpy-elementwise-outer-product
                # sigma = np.sum(r_div_q_gamma[:,:,None] * diff**2,axis=0)/\
                #     np.sum(r_div_q_gamma,axis=0,keepdims=True).T
                # sigma = np.apply_along_axis(np.diag,axis=1,arr=sigma) + covar_regularization * np.eye(X.shape[-1])
                
                # product,posPart,negPart = splitExpSumLog(diff**2,log_r_div_q_gamma[:,:,None])
                # try:
                #     sigma = np.sum(np.exp(posPart - np.repeat(log_sum_r_div_q_gamma.T,posPart.shape[-1],axis=1)).filled(0) -\
                #                 np.exp(negPart - np.repeat(log_sum_r_div_q_gamma.T,posPart.shape[-1],axis=1)).filled(0),axis=0)
                #     sigma = np.apply_along_axis(np.diag,axis=1,arr=sigma) + covar_regularization * np.eye(X.shape[-1])
                     
                # except Exception as e:
                #     raise e
                
                sigma = np.sum(r_div_q_gamma_normed[:,:,None] * diff**2,axis=0)
                sigma = np.apply_along_axis(np.diag,axis=1,arr=sigma) + covar_regularization * np.eye(X.shape[-1])
                
                if len(sigma.shape)==2:
                    sigma = sigma[None,:,:]
        
        sigma = self.spectralClipping(sigma)
        
        # Check that there is no inf/nan values and that sigma is not singular
        try:
            assert(not np.any(np.isnan(sigma)))
            assert(not np.any(sigma==np.inf))
            
            # if self.covar_regularization > 0:
            #     for i in range(sigma.shape[0]):
            #         assert(np.linalg.det(sigma[i].astype(np.float64))!=0)
        except Exception as e:
            print('Floating point error while computing sigma!')
            for i in range(sigma.shape[0]):
                print(sigma[i])
            raise e   
         
        return alpha,mu,sigma
    
    def getMCE(self,params):
        """
        Get the MCE in the case where we are fitting a multivariate Gaussian
        (not a GMM)

        Parameters
        ----------
        params : TYPE
            DESCRIPTION.

        Raises
        ------
        e
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        
        covar_regularization = self.covar_regularization #Add ridge to covar matrix to prevent numerical instability
        
        # To make this code (slightly) more readable

        # X,q = np.concatenate(self.X_gmm,axis=0),self.q
        X,q = np.concatenate(self.X,axis=0),self.q
        
        log_r_div_q = np.concatenate(self.Hx_WxList,axis=0)
        
        # Compute new alpha, mu
        log_sum_r_div_q = logsumexp(log_r_div_q)
        
        # If some weights are infinite, weight the corresponding X samples equally
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            infInd = np.isinf(np.exp(log_r_div_q))
        if np.any(infInd):
            weights = np.zeros_like(log_r_div_q)
            weights[infInd] = 1/np.sum(infInd)
        else:
            weights = np.exp(log_r_div_q - log_sum_r_div_q)
            
        mu = np.sum(weights * X,axis=0)
        diff = X - mu
        Sigma = np.sum(weights[:,:,None] * diff[:,None,:] * diff[:,:,None],axis=0)
        
        Sigma = self.spectralClipping(Sigma[None,:,:]).squeeze()
        
        return np.ones((1,)),mu[None,:],Sigma[None,:,:]
    
    def updateParams(self,params,eps=1e-2,condThresh=1e5,maxiter=10,retCE=False):
        """
        Update params by performing EM iterations until converged
    
        Parameters
        ----------
        initParams: GMMParams
            The initial params of the GMM to be fit with EM.
        condThresh : float, optional
            The threshold for the condition number of the covar matrix beyond
            which we abort EM. Default is 1e5.
        eps : TYPE, optional
            DESCRIPTION. The default is 10**-3.
        maxiter : int, optional
            The maximum number of EM steps to perform. Default is 10.
    
        Returns
        -------
        params : GMMParams
            The new updated params.
    
        """
        i = 0        
        converged  = False
        ce_old = np.inf
        
        if params.k() > 1:
            # Loop until EM converges
            while not converged and i < maxiter:
                
                # Perform a single EM iteration
                alpha,mu,sigma = self.emIteration(params)
                
                # If alpha is -1, we had an error taking log likelihood
                if type(alpha)==int and alpha==-1:
                    return params,np.inf
                
                # Check if the EM iteration resulted in alpha=0
                if type(alpha)!=np.ndarray:
                    if alpha == -1:
                    
                        if retCE:
                            return -1,None
                        else:
                            return -1
                
                # Check to make sure that the new covar matrices are well-conditioned
                condNum = np.linalg.cond(sigma.astype(np.float64))
                if np.any(condNum>condThresh):
                    
                    # Let's check if the the GMM with one component is concentrating on one observation
                    if params.k() < 2:
                        X = np.concatenate(self.X,axis=0)
                        dist = np.linalg.norm(X-mu,axis=0)
                        concentrated = np.sum(dist==0)>0
                        
                        if concentrated:
                            print('The GMM has concentrated on a single observation!')
                    
                    # If covar is not well conditioned (i.e. model is overfit), abort the EM procedure
                    if not retCE:
                        return None
                    else:
                        errStr = ''
                        if params.k()==1:
                            numEventsOcc = np.zeros(len(self.X))
                            for i in range(len(self.X)):
                                numEventsOcc[i] = np.sum(self.rList[i]>0)
                            kList = [_params.k() for _params in self.paramsList]                                
                            errStr = 'condition number of Sigma: {} \n'.format(condNum)+\
                                'Number of obs. which correspond to event occuring at each stage:{} \n'.format(numEventsOcc)+\
                                'k selected at each stage: {} \n'.format(kList)
                        return errStr,None
                
                # Update the params using the result of single EM iteration
                params.update(alpha,mu,sigma)
                
                # Compute cross-entropy for the updated params
                # ce = self.c_bar(params,gmm=True)
                try:
                    ce = self.c_bar(params)
                except RuntimeWarning as e:
                    
                    # If this happens, we just consider it a failure
                    if str(e)=='overflow encountered in double_scalars' or\
                        str(e)=='overflow encountered in square':
                        if retCE:
                            return -1,None
                        else:
                            return -1
                    else:
                        raise e
                
                # Compute the change in cross-entropy
                converged = (ce_old - ce) < eps*np.abs(ce_old)
                ce_old = ce
                i += 1
        else:
            
            alpha,mu,sigma = self.getMCE(params)
            
            # Update the params using the result of single EM iteration
            params.update(alpha,mu,sigma)
            
            # Compute cross-entropy for the updated params
            try:
                ce = self.c_bar(params)
            except Exception as e:
                print(e)
                raise e

        if not retCE:
            return params
        else:
            return params,ce
    
    def runStage(self,s,params,kMin,kMax=30,numInitParams=10):
        """
        Run a single stage of the CEM procedure for each value of k considered.

        Parameters
        ----------
        s : int
            The stage number.
        params : GMMParams
            Previous best GMMParams to draw new samples from.
        kMin: int
            The smallest value of k considered
        numInitParams : int, optional
            The number of initial params used in the EM alg.

        Returns
        -------
        bestParamsList : list
            The best GMMParams (in terms of cross-entropy) for each value of k
        cicList : list
            The CIC for each GMMParams.
        cicMA : list
            The moving average of the cicList
        """
        
        # Determine kMax from number of samples and computing num free params 
        if s > 0:
            X = np.concatenate(self.X,axis=0)
            max_free_param = np.floor(X.shape[0] / 10) #maximum number of free parameters we would like to allow.  If the divisor is 10, it means we expect that each component has 10 observations on average. 
            if self.covarStruct=='homogeneous':
                kMax = min(kMax,np.floor((max_free_param+1)/(X.shape[1] + 2)))
            elif self.covarStruct=='diagonal':
                kMax = min(kMax,np.floor((max_free_param+1)/(2*X.shape[1] + 1)))
            elif self.covarStruct=='full':
                kMax = min(kMax,np.floor((max_free_param+1)/(X.shape[1] + (X.shape[1]*(X.shape[1]+1))/2 + 1)))
        
        # Draw samples from previous best GMMParams and from nominal density
        if not self.variableSampleSizes:
            nSamples = self.sampleSize
            
        else:
            nSamples = self.sampleSize[s]
        
        if self.seis:
            nominalSamples = int(np.ceil(nSamples * self.alpha))
            gmmSamples = nSamples - nominalSamples
            
            xNominal = self.generateX(None,nominalSamples)
            xGmm = self.generateX(params,gmmSamples)
            
            x = np.concatenate([xNominal,xGmm],axis=0)
            
        else:
            xGmm = self.generateX(params,nSamples)
            x = xGmm.copy()
        
        # Add new samples to existing samples
        if s==0:
            self.X = [x,]
            self.X_gmm = [xGmm,]
        else:
            self.X.append(x)
            self.X_gmm.append(xGmm)
        
        # Run simulation and compute likelihood ratio
        try:
            # r_x = self.r(x)
            
            # Get simulation result
            Hx = self.h(x)
            # Get likelihood ratio
            log_q_theta = self.log_q(params)
            log_px = self.p(x) # this actually computes the log density
            log_qx = log_q_theta(x)
            
            # self.pxList.append(np.exp(log_px))
            self.logpxList.append(log_px)
            # self.qxList.append(np.exp(log_qx))
            
            # Compute SEIS mixture
            log_qx_mix = log_qx#self.alpha * np.exp(log_px) + (1-self.alpha) * np.exp(log_qx)
            
            self.logqxList.append(log_qx_mix)
            
            # gmmStIndex = int(np.ceil(nSamples * self.alpha))
            # log_Wx = log_px[gmmStIndex:] - log_qx[gmmStIndex:]
            log_Wx = log_px - log_qx_mix
            
            # try:
            #     assert(not np.any(Wx==np.inf))
            # except:
            #     print('Overflow in exponential!')
            #     print(np.max(log_px - log_qx))
            #     raise Exception
            
            r_x = Hx * np.exp(log_px)
            
            # This computation involves multiplying a likelihood ratio by 0 or 1. 
            # Shouldn't be any numerical issues here.
            
            # Need to catch warnings here because we may take a log of zero
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # log_Hx_Wx = np.log(Hx[gmmStIndex:]) + log_Wx
                log_Hx_Wx = np.log(Hx) + log_Wx
                
            # _,_,qx_alpha = defensiveIS.getBeta2Mix(Hx,self.pxList[-1],self.qxList[-1],self.alpha)
            # Hx_Wx = np.exp(np.log(self.pxList[-1])-np.log(qx_alpha))
            
        except Exception as e:
            if self.log:
                logging.error('Error during r(x): {}'.format(str(e)))
            raise e         

        self.rList.append(r_x)
        self.hList.append(Hx)
        self.Hx_WxList.append(log_Hx_Wx)
        
        # Make sure we have enough non-zero Hx values that we can account for all DOF in GMM.
        if s == 0:
            Hx_Wx_total = np.concatenate(self.Hx_WxList, axis=0)    
        else:
            Hx_Wx_total = np.concatenate(self.Hx_WxList[1:], axis=0)    
        
        # If this occurs, we do not have enough LI vectors to estimate the GMM params
        numUsableObs = np.sum(1-np.isinf(Hx_Wx_total))
        if numUsableObs < params.dim() + 3:
            return None,None,None
        
        # Otherwise, try to run the EM algo
        else:
            
            maxInitTries = 5
            for attempt in range(maxInitTries):
                results = self.emFit(s,kMin,kMax,numInitParams,params,numUsableObs)
                
                # Check to make sure the EM algo worked
                if results[0] is not None:
                    return results
                
        raise(Exception('Unable to fit any GMM to the data!'))
            
            
        
    def emFit(self,s,kMin,kMax,numInitParams,params,numUsableObs):
        """
        Fit the GMM to the data using the EM algorithm.

        Parameters
        ----------
        s : TYPE
            DESCRIPTION.
        kMin : TYPE
            DESCRIPTION.
        kMax : TYPE
            DESCRIPTION.
        numInitParams : TYPE
            DESCRIPTION.
        params : TYPE
            DESCRIPTION.
        numUsableObs : TYPE
            DESCRIPTION.

        Raises
        ------
        e
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        # Run EM for different values of k to find best GMM fit
        k = kMin
        kTooBig = False
        retry = True
        maIncr = False
        resetCounter = 0
        runStageCounter = 0
        
        # Window size for computing CIC moving average
        windowSize=4
        
        bestParamsList = []
        cicList = []
        cicMA = []
    
        # Keep increasing k until we abort more than 50% of EM trials
        while not kTooBig and not maIncr and k <= kMax:
            
            if self.verbose:
                print('k={}'.format(k))
            
            # Define the intial params used for each EM trial
            initParamsList = self.sampleInitParams(s, numInitParams, k, params.dim())
            updatedParamsList = [None,]*numInitParams
            ceArray = np.ones(shape=(numInitParams,1)) * np.inf
            
            # Run the EM trials for current value of k
            for i in range(len(initParamsList)):
                try:
                    updatedParams,ce = self.updateParams(initParamsList[i],retCE=True)
                except Exception as e:
                    if self.log:
                        logging.error('Error while updating params: {}'.format(str(e)))
                    raise e
                    
                # If the EM got one or more mixtures equal to zero, break out of the loop
                if updatedParams == -1:
                    kTooBig = True
                    retry = False
                    break
            
                updatedParamsList[i] = updatedParams
                # If ce is None, then EM was terminated. Set ce to infinity.
                ceArray[i] = ce if ce is not None else np.inf
                
            # Check to see if more than 50% of EM trial were terminated or CIC moving avg is increasing
            if not kTooBig:
                kTooBig = np.sum(ceArray==np.inf)>(numInitParams//2)
            
            # If k is too big and k > kMin, stop increasing k and return
            if kTooBig and k > kMin:
                    break
            
            # If k == kMin
            elif kTooBig and k == kMin:
            
                # If all of the EM attempts failed, decrease kMin and try again
                if np.all(ceArray==np.inf) and retry:
                    
                    # Run functions to print em results
                    for errStr in updatedParamsList:
                        print(errStr)
                    
                    # In this case, reduce kMin by one and re-attempt
                    kMin = max(kMin-1,1)
                    nextk = kMin
                    
                    # Need to reset kTooBig to False here
                    kTooBig = False
                    
                    resetCounter += 1
                    
                    print('In stage: {}'.format(s))
                    print('Num k={} resets: {}'.format(k,resetCounter))
                    
                    # If we have too many failures, stop the program. Otherwise this can be an
                    # endless loop
                    if resetCounter > 100:
                        e = Exception('Unable to solve EM after 100 tries...aborting (In stage {} w/ numUsableObs={})!'.format(s,numUsableObs))
                        raise(e)
                
                # If only some of them failed, just take what we've got and break
                elif not np.all(ceArray==np.inf):
                        
                    # Choose the best result of all initial params for the current value of k
                    # Choose the best params with lowest cross-entropy
                    bestParamsInd = np.argmin(ceArray)
                    bestParams = updatedParamsList[bestParamsInd]
                    
                    # This will occur if all of the update attempts failed
                    if bestParams is None:
                        break
                    
                    # Add new params to the list
                    bestParamsList.append(bestParams)         
        
                    # Compute CIC for the best params
                    try:
                        cic = self.cic(bestParams)
                    except Exception as e:
                        if self.log:
                            print('Error computing CIC during runStage!')
                            logging.error('Error while computing CIC: {}'.format(str(e)))
                        raise e
                        
                    cicList.append(cic)
                    
                    resetCounter = 0
                    
                    # Break after doing these computations
                    break
                
                # This will occur if k is determined to be too big
                else:
                    break
                    
            # If k isn't too big, increment for the next loop
            else:
                nextk = k+1
                
                # Choose the best result of all initial params for the current value of k
                # Choose the best params with lowest cross-entropy
                bestParamsInd = np.argmin(ceArray)
                bestParams = updatedParamsList[bestParamsInd]
                
                # Add new params to the list
                bestParamsList.append(bestParams)         
    
                # Compute CIC for the best params
                try:
                    cic = self.cic(bestParams)
                except Exception as e:
                    if self.log:
                        logging.error('Error while computing CIC: {}'.format(str(e)))
                    raise e
                    
                cicList.append(cic)
                
                # If k is large enough, compute the moving average of past few CIC values
                if  k >= kMin+windowSize-1:
                    ma = sum(cicList[k-windowSize:k-1])
                    
                    if len(cicMA) > 0:
                        # If the moving averaged has increased, time to stop
                        maIncr = (ma > cicMA[-1])
                
                    cicMA.append(ma)
            
                resetCounter = 0
            
            # Increment k 
            k = nextk
            # runStageCounter += 1
            # print('iterations in this stage: {}'.format(runStageCounter))
        
        # If all of the initial parameters resulted in a failed EM alg., return None
        try:
            assert(len(cicList)>0)
        except Exception as e:
            print('No CIC values were computed!')
            print('kMin: {}'.format(kMin))
            print('kMax: {}'.format(kMax))
            print('Number of CIC values computed: {}'.format(len(cicList)))
            print('Current k value: {}'.format(k))
            print()
            print(e)
            
            return None,None,None
            
        try:
            assert(len(cicList)==len(bestParamsList))
        except Exception as e:
            print('Number of CIC values does not match the number of parameters!')
            print()
            print(e)
            
        return bestParamsList,cicList,cicMA
    
    