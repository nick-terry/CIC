#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 13:25:44 2020

@author: nick
"""

'''

Implementation of cross-entropy information criterion method from the paper
"Information Criterion for Boltzmann Approximation Problems" by Choe, Chen, and Terry.

'''
import numpy as np
# Want to raise an exception if divide by zero
# np.seterr(divide='raise')
import scipy.stats as stat
import scipy.special as spc
import pickle as pck
import datetime as dttm
import os
import copy
import logging
import warnings

# warnings.simplefilter("error")
    
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
    
    def dim(self):
        return self._dataDim
    
    def get(self):
        return self._alpha,self._mu,self._sigma
    
    def update(self,alpha,mu,sigma):
        
        # Check all the dims
        try:
            assert(alpha.shape[0]==mu.shape[0])
            assert(mu.shape[1]==self._dataDim)
            assert(sigma.shape[0]==alpha.shape[0])
            assert(sigma.shape[1]==self._dataDim and sigma.shape[2]==self._dataDim)
        except Exception as e:
            print(e)
            raise(e)
        
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
    
class CEM:
    
    def __init__(self,initParams,p,h,**kwargs):
        """
        Create a new CEM (Cross-Entropy Minimizer)

        Parameters
        ----------
        initParams : GMMParams
            The initial parameters of the GMM model.
        p : function
            Returns the true density (likelihood) at a given point.
        h : function
            Simulator evaluation
        **kwargs : dict
            Keyword arguments.

        """
        
        self.initParams = initParams
        self.paramsList = [initParams,]
        self.dim = initParams.get()[1].shape[1]
        self.p = p
        self.h = h
        
        self.hList = []
        self.rList = []
        self.Hx_WxList = []
        
        # Store some historical data as procedure is executed
        self.bestParamsLists = []
        self.cicLists = []
        self.s = 0
        
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
            
        self.covar_regularization = 10**-100
            
        self.cicArray = np.zeros(shape=(self.numIters,1))
            
    def getVectorizedDensity(self,densityFn):
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
        Hx_Wx = np.exp(np.concatenate(self.Hx_WxList))
        c_bar2 = -np.sum(Hx_Wx * llh)/n
        
        # Use b trick
        Hx_Wx = np.concatenate(self.Hx_WxList)
        
        llh_gtz = llh > 0
        llh_p,llh_n = llh[llh_gtz],-llh[~llh_gtz]
        
        posX = Hx_Wx[llh_gtz]+np.log(llh_p)-np.log(n)
        negX = Hx_Wx[~llh_gtz]+np.log(llh_n)-np.log(n)
        
        # Get magic constant b
        # mp = np.max(posX) if posX.size>0 else -np.inf
        # mn = np.max(negX) if negX.size>0 else -np.inf
        # b = max(mp,mn)
        
        # log_c_bar = b + np.log(-(np.sum(np.exp(posX-b)) -\
        #                          np.sum(np.exp(negX-b))))
        
        if posX.size > 0: 
            bp = np.max(posX)
            lp = bp + spc.logsumexp(posX-bp)
        else:
            lp = -np.inf
        
        if negX.size > 0:
            bn = np.max(negX)
            ln = bn + spc.logsumexp(negX-bn)
        else:
            ln = -np.inf
        
        # Try to exponentiate the log values for finishing calculation
        np.seterr(over='raise')
        
        if np.isinf(lp):
            pos = 0
            
        else:
            try:
                pos = np.exp(lp.astype(np.float128))
            except Exception as e:
                print('Overflow during CE computation! (positive part')
                print(lp)
                raise e
        
        if np.isinf(ln):
            neg = 0
            
        else:
            try:
                neg = np.exp(ln.astype(np.float128))
            except Exception as e:
                print('Overflow during CE computation! (negative part)')
                print(ln)
                raise e
            
        np.seterr(over='warn')
        
        c_bar = -(pos - neg)
            
        return c_bar
    
    def rho(self):
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
        if len(self.X)==1:
            _rho = np.mean(np.concatenate(self.Hx_WxList,axis=0))
        
        # Otherwise, use cumulative estimate that excludes zeroth stage
        else:
            _rho = np.mean(np.concatenate(self.Hx_WxList[1:],axis=0))
            
        return _rho
    
    def cic(self,params):
        """
        Compute the cross-entropy information criterion (CIC) defined in equation 3.13 
        of the paper. This is a more general implementation which does not specify
        the relationship between eta and theta.
    
        Parameters
        ----------
        params : GMMParams
            Params for which CIC is computed.
        
        Returns
        -------
        _cic : float
            The CIC.
    
        """
        X = np.concatenate(self.X,axis=0)
        HxWx = np.concatenate(self.Hx_WxList)
        # n = HxWx.shape[0] - np.sum(np.isinf(HxWx))
        n = X.shape[0]
        
        # Compute dimension of model parameter space from the number of mixtures, k
        k = params.k()
        p = X.shape[1]
        d = (k-1)+k*(p+p*(p+1)/2)
        
        _cic = self.c_bar(params) + self.rho()*d/n

        return _cic
    
    def r(self,x):
        
        # Run simulation
        h_x = self.h(x)
        
        # Compute probability of contingency vector given current parameters
        density = self.p(x)
        
        return h_x*density
    
    def q(self,params):
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
            
            # Concatenate samples from each stage to a single numpy array
            if type(X) is list:
                _X = np.concatenate(X,axis=0)
            else:
                _X = X
                
            _alpha = np.tile(alpha,(_X.shape[0],1))
            
            # Compute density at each observation
            densities = np.zeros(shape=(_X.shape[0],params.k()))
            for j in range(params.k()):
                densities[:,j] = stat.multivariate_normal.pdf(_X,mu[j,:],sigma[j,:],allow_singular=self.allowSingular)
            
            densities = np.expand_dims(np.sum(np.exp(np.log(_alpha) + np.log(densities)),axis=1),axis=1)
            
            return densities
        
        return _q
    
    def log_q(self,params):
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
        
        def _log_q(X):
        
            if type(X) is list:
                _X = np.concatenate(X,axis=0)
            else:
                _X = X
                
            _alpha = np.tile(alpha,(X.shape[0],1))
            
            # Compute density at each observation
            log_densities = np.zeros(shape=(_X.shape[0],params.k()))
            try:
                for j in range(params.k()):
                    try:
                        log_densities[:,j] = stat.multivariate_normal.logpdf(_X,mu[j,:],sigma[j,:],allow_singular=self.allowSingular)
                    except Exception as e:
                        raise(e)
                
                log_densities = np.expand_dims(spc.logsumexp(np.log(_alpha)+log_densities,axis=1),axis=1)
            
            except np.linalg.LinAlgError as e:
                print('Singular covariance matrix in the GMM!')
                print(e)
                self.errored = True
            
            return log_densities
        
        return _log_q
    
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
    
    def expectation(self,x,q,params):
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
        _log_alpha = np.tile(np.log(alpha),(x.shape[0],1))
        
        # Compute density at each observation
        log_densities = np.zeros(shape=(x.shape[0],params.k()))
        for j in range(params.k()):
            try:
                log_densities[:,j] = stat.multivariate_normal.logpdf(x,mu[j,:],sigma[j,:],allow_singular=self.allowSingular)
            except Exception as e:
                print(sigma[j,:])
                raise(e)
                
        log_alpha_q = _log_alpha + log_densities

        log_density = np.expand_dims(spc.logsumexp(log_alpha_q,axis=1),axis=1)
        _log_density = np.tile(log_density,(1,params.k()))
        
        gamma = np.exp(log_alpha_q - _log_density)
        
        # For each i (observation), summing gamma over j should give 1
        try:
            err = np.abs(np.sum(gamma,axis=1)-1)
            assert(np.all(err<1e-3))
        except Exception as e:
            print(params.k())
            print('Gammas don\'t sum to one for at least one observation!')
            raise e
        
        return gamma
    
    def log_expectation(self,x,q,params):
        """
        Expectation step of EM algorithm. Returns log of gamma.
    
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
        _log_alpha = np.tile(np.log(alpha),(x.shape[0],1))
        
        # Compute density at each observation
        log_densities = np.zeros(shape=(x.shape[0],params.k()))
        for j in range(params.k()):
            try:
                log_densities[:,j] = stat.multivariate_normal.logpdf(x,mu[j,:],sigma[j,:],allow_singular=self.allowSingular)
            except Exception as e:
                print(sigma[j,:])
                raise(e)
                
        log_alpha_q = _log_alpha + log_densities

        log_density = np.expand_dims(spc.logsumexp(log_alpha_q,axis=1),axis=1)
        _log_density = np.tile(log_density,(1,params.k()))
        
        log_gamma = log_alpha_q - _log_density
        
        # Check that nothing weird happened
        try:
            assert(not np.any(np.isnan(log_gamma)))
            assert(not np.any(np.isinf(log_gamma)))
        except Exception as e:
            # If this happens,consider the EM update to have failed
            print('Error getting log likelihood of GMM component!')
            raise e
        
        return log_gamma
    
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
        X,q = np.concatenate(self.X,axis=0),self.q
         
        # gamma = self.expectation(X, q, params)
        # Added small amount of regularization here to prevent a component having alpha=0
        # TODO: See if there is a better solution to this
        log_gamma =  self.log_expectation(X, q, params) + self.covar_regularization
        
        
        r_div_q = np.concatenate(self.Hx_WxList,axis=0)
        
        # We need to avoid taking log of zero entries here
        r_div_q_gamma = np.zeros_like(log_gamma)
        nzi = np.squeeze(r_div_q>0) 
        
        # try:
        #     assert(np.all(gamma[nzi]>0))
        # except Exception as e:
        #     print('Some gamma values are negative!')
        #     raise e
        
        # This works since multiplying by the zero entries still yields zero
        r_div_q_gamma[nzi] = np.exp( np.log(r_div_q[nzi]) + log_gamma[nzi] )
        
        # Compute new alpha, mu
        # TODO: Adjust how alpha is computed so no mixture components go to zero. May
        alpha = np.exp(np.log(np.sum(r_div_q_gamma,axis=0)) - np.log(np.sum(r_div_q)))
        
        # Check that the mixing proportions sum to 1
        try:
            assert(np.abs(np.sum(alpha)-1)<1e-3)
        except Exception as e:
            if self.log:
                logging.warning('alpha computed during M step does not sum to one! Normalizing alpha...')
            alpha = alpha/np.sum(alpha)
        
        # Check that there is no inf/nan values
        try:
            assert(not np.any(np.isnan(alpha)))
            assert(not np.any(alpha==np.inf))
        except Exception as e:
            print('Floating point error while computing alpha!')
            raise e

        mu = np.sum(r_div_q_gamma[:,:,None]*X[:,None,:],axis=0)/np.sum(r_div_q_gamma,axis=0,keepdims=True).T
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
                sigma = np.sum(r_div_q_gamma[:,:,None]*(diff[:,:,None]*diff[:,None]),axis=0)/\
                    np.sum(r_div_q_gamma,axis=0) + covar_regularization * np.eye(mu.shape[-1])
                if len(sigma.shape)==2:
                    sigma = sigma[None,:,:]
            
            else:
                # Compute difference of data from mean vector
                diff = X[:,None,:]-mu
                # Compute sigma using a vectorized outer product. 
                # See https://stackoverflow.com/questions/42378936/numpy-elementwise-outer-product
                sigma = np.sum(r_div_q_gamma[:,:,None,None]*(diff[:,:,:,None]*diff[:,:,None]),axis=0)/\
                    np.sum(r_div_q_gamma,axis=0,keepdims=True).T[:,:,None] + covar_regularization * np.eye(mu.shape[-1])
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
                sigma = np.sum(r_div_q_gamma*np.sum(diff**2,axis=-1,keepdims=True),axis=0)[:,None]/\
                    np.sum(r_div_q_gamma,axis=0)/X.shape[-1]
                sigma = (sigma + covar_regularization) * np.eye(X.shape[-1])
                if len(sigma.shape)==2:
                    sigma = sigma[None,:,:]
        
            else:
                # Compute difference of data from mean vector
                diff = X[:,None,:]-mu
                # Compute sigma using a vectorized outer product. 
                # See https://stackoverflow.com/questions/42378936/numpy-elementwise-outer-product
                sigma = np.sum(r_div_q_gamma[:,:,None]*np.sum(diff**2,axis=-1,keepdims=True),axis=0)/\
                    np.sum(r_div_q_gamma,axis=0,keepdims=True).T/X.shape[-1] + covar_regularization
                sigma = np.repeat(np.repeat(sigma[:,:,None],axis=1,repeats=X.shape[-1]),
                                  axis=2,repeats=X.shape[-1]) * np.eye(X.shape[-1])
                if len(sigma.shape)==2:
                    sigma = sigma[None,:,:]
            
        elif self.covarStruct=='diagonal':
            if mu.shape[0]==1:
                # Compute difference of data from mean vector
                diff = X-mu
                # Compute sigma using a vectorized outer product. 
                # See https://stackoverflow.com/questions/42378936/numpy-elementwise-outer-product
                sigma = np.sum(r_div_q_gamma * diff**2,axis=0)/\
                    np.sum(r_div_q_gamma,axis=0)/X.shape[-1]
                sigma = np.diag(sigma) + covar_regularization * np.eye(X.shape[-1])
                if len(sigma.shape)==2:
                    sigma = sigma[None,:,:]
        
            else:
                # Compute difference of data from mean vector
                diff = X[:,None,:]-mu
                # Compute sigma using a vectorized outer product. 
                # See https://stackoverflow.com/questions/42378936/numpy-elementwise-outer-product
                sigma = np.sum(r_div_q_gamma[:,:,None] * diff**2,axis=0)/\
                    np.sum(r_div_q_gamma,axis=0,keepdims=True).T
                sigma = np.apply_along_axis(np.diag,axis=1,arr=sigma) + covar_regularization * np.eye(X.shape[-1])
                if len(sigma.shape)==2:
                    sigma = sigma[None,:,:]
            
        # Check that there is no inf/nan values and that sigma is not singular
        try:
            assert(not np.any(np.isnan(sigma)))
            assert(not np.any(sigma==np.inf))
            for i in range(sigma.shape[0]):
                assert(np.linalg.det(sigma[i].astype(np.float64))!=0)
        except Exception as e:
            print('Floating point error while computing sigma!')
            raise e   
         
        return alpha,mu,sigma
    
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
        
        # Loop until EM converges
        while not converged and i < maxiter:
            
            # Perform a single EM iteration
            alpha,mu,sigma = self.emIteration(params)
            
            # Check to make sure that the new covar matrices are well-conditioned
            condNum = np.linalg.cond(sigma.astype(np.float64))
            if np.any(condNum>condThresh):
                
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
            ce = self.c_bar(params)
            
            # Compute the change in cross-entropy
            converged = (ce_old - ce) < eps*np.abs(ce_old)
            ce_old = ce
            i += 1

        if not retCE:
            return params
        else:
            return params,ce
        
    def sampleInitParams(self,stage,numInitParams,k,dim=None):
        """
        Get the initial params used to fit the GMM using EM.

        Parameters
        ----------
        stage : int
            The stage of the CEM procedure.
        numInitParams : int
            The number of initial params to create.
        k : int
            The number of components in each GMM.
        dim : int
            The dimension of the MVG dist. used to create the GMM. Must be specified if stage==0

        Returns
        -------
        initParamsList : List
            The list of all intial params generated.

        """
        
        #TODO : fix having not homog covar initially w/ no samples
        initParamsList = []
        
        # At stage zero, randomly choose all GMM params by drawing from standard MVG dist.
        if stage==0:
            
            try:
                assert(dim is not None)
            except Exception as e:
                print('Dim must be defined for creating initial params in the first stage!')
                raise e
                
            for i in range(numInitParams):
                # Initial guess for GMM params
                alpha0 = np.ones(shape=(k,))/k
                
                # Randomly intialize the means of the Gaussian mixture components
                mu0 = np.random.multivariate_normal(np.zeros(dim),
                                                    np.eye(dim),
                                                    size=k)
                
                # Set covariance matrix to be identity
                sigma0 = 3 * np.repeat(np.eye(dim)[None,:,:],k,axis=0)
                params = GMMParams(alpha0, mu0, sigma0, dim)
                
                initParamsList.append(params)
        
        # If stage > 0, we create half of init params from MVG and half of init params by sampling from observations from s=1,...,t
        else:
            
            # Get indices where Hx>0
            posInd = np.where(np.concatenate(self.hList,axis=0))[0]
            zeroInd = np.where(1-np.concatenate(self.hList,axis=0))[0]
            X = np.concatenate(self.X,axis=0)
            
            # Get half of init params by sampling from the MVG
            numMVGParams = 0 #numInitParams//2
            
            try:
                assert(dim is not None)
            except Exception as e:
                print('Dim must be defined for creating initial params!')
                raise e
                
            for i in range(numMVGParams):
                # Initial guess for GMM params
                alpha0 = np.ones(shape=(k,))/k
                
                # The mean for the MVG is chosen to be the mean of all observations where h>0
                selectedData = X[posInd,:]
                # print('shape of mean of event data: {}'.format(selectedData.shape))
                muPosH = np.mean(selectedData,axis=0)
                
                # Make sure nothing funny happened when computing this mean
                try:
                    assert(muPosH.shape[0]==self.X.shape[-1])
                except Exception as e:
                    print('Computing mean observation failed somehow!')
                    if self.log:
                        logging.ERROR('Computing mean observation failed somehow!')
                    raise e
                
                # Randomly intialize the means of the Gaussian mixture components
                mu0 = np.random.multivariate_normal(muPosH,
                                                    np.eye(dim),
                                                    size=k)
                
                # Set covariance matrix to be identity
                sigma0 = 3 * np.repeat(np.eye(dim)[None,:,:],k,axis=0)
                params = GMMParams(alpha0, mu0, sigma0, dim)
                
                initParamsList.append(params)
              
            # For the remaining initParams, randomly select k samples w/o replacement. Prefer positive entropy samples.
            numChoices = posInd.shape[0]
            for i in range(numInitParams-numMVGParams):
                
                if numChoices>0:
                    choice = np.random.choice(range(numChoices),size=min(k,numChoices),replace=False).astype(int)
                    chosenInd = list(posInd[choice])
                else:
                    chosenInd = []
                
                # If there was not enough samples w/ positive entropy, draw some with 0 entropy
                if len(chosenInd) < k:
                    
                    # Randomly select remaining needed samples w/o replacement
                    choice = np.random.choice(range(len(zeroInd)),size=k-len(chosenInd),replace=False).astype(int)
                    
                    if zeroInd[choice].size == 1:
                        chosenInd.append(zeroInd[choice])
                    else:
                        chosenInd += list(zeroInd[choice])
                    
                try:
                    chosenInd = np.array(chosenInd).astype(int)
                except Exception as e:
                    print(chosenInd)
                    raise e
                    
                # Create XBar matrix by concatenating the data vectors
                # Get the sampled data
                XBar = X[chosenInd,:]
                
                # Compute covar matrix for GMM params
                p = X.shape[-1]
                covar = np.cov(XBar,rowvar=False)
                tr = np.trace(covar) if len(covar.shape)>0 else covar
                
                covar = 3/p*tr*np.eye(p) + np.eye(p) * self.covar_regularization
                
                # check that we are not somehow outputting zero matrix
                try:
                    assert(not np.all(covar==0))
                except:
                    print('Init params for the GMM has a degenerate covar matrix (all zeros)')
                    print(XBar)
                    
                    covar = 3/p*np.eye(p)
                
                covar = np.tile(np.expand_dims(covar,axis=0),(k,1,1))
                
                # In case the trace is so small that we get some numerical issues in EM
                covar = self.spectralClipping(covar)
                
                # Create GMMParams object and add to list
                initParams  = GMMParams(np.ones(k)/k, XBar, covar, p)
                initParamsList.append(initParams)
            
            
            
        return initParamsList
     
    def spectralClipping(self,sigma):
        # Apply the spectral heuristic from Coretto and Hennig, 2016
        
        # Get eigen decomposition of covar matrices
        v,W = np.linalg.eig(sigma.astype(np.float64))
        lambda_max = np.max(v)
        
        # Set each eigenvalue lambda to be max(lambda_max,lambda)
        v = np.maximum(v,lambda_max/self.gamma*np.ones_like(v))
        
        # Reconstruct covar matrices from the decomposition
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        
            # Suppress warnings for 
            for k in range(sigma.shape[0]):
                sigma[k] = (W[k] @ np.diag(v[k]) @ np.linalg.inv(W[k])).astype(np.float128)
        
        return sigma
    
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
        
        # Window size for computing CIC moving average
        windowSize=4
        
        # Draw samples from previous best GMMParams
        if not self.variableSampleSizes:
            x = self.generateX(params,self.sampleSize)
        else:
            x = self.generateX(params,self.sampleSize[s])
        
        # Add new samples to existing samples
        if s==0:
            self.X = [x,]
        else:
            self.X.append(x)
        
        # Run simulation and compute likelihood ratio
        try:
            # r_x = self.r(x)
            
            # Get simulation result
            Hx = self.h(x)
            # Get likelihood ratio
            log_q_theta = self.log_q(params)
            log_px = self.p(x) # this actually computes the log density
            Wx = np.exp(log_px - log_q_theta(x))
            
            r_x = Hx * np.exp(log_px)
            
            # This computation involves multiplying a likelihood ratio by 0 or 1. 
            # Shouldn't be any numerical issues here.
            Hx_Wx = Hx * Wx
            
        except Exception as e:
            if self.log:
                logging.error('Error during r(x): {}'.format(str(e)))
            raise e
            
        self.rList.append(r_x)
        self.hList.append(Hx)
        self.Hx_WxList.append(Hx_Wx)
        
        bestParamsList = []
        cicList = []
        cicMA = []
        
        # Run EM for different values of k to find best GMM fit
        k = kMin
        kTooBig = False
        maIncr = False
        resetCounter = 0
        runStageCounter = 0
    
        # Keep increasing k until we abort more than 50% of EM trials
        while not kTooBig and not maIncr and k <= kMax:
            
            if self.verbose:
                print('k={}'.format(k))
            
            # Define the intial params used for each EM trial
            initParamsList = self.sampleInitParams(s, numInitParams, k, params.dim())
            updatedParamsList = []
            ceArray = np.zeros(shape=(numInitParams,1))
            
            # Run the EM trials for current value of k
            for i in range(len(initParamsList)):
                try:
                    updatedParams,ce = self.updateParams(initParamsList[i],retCE=True)
                except Exception as e:
                    if self.log:
                        logging.error('Error while updating params: {}'.format(str(e)))
                    raise e
            
                updatedParamsList.append(updatedParams)
                # If ce is None, then EM was terminated. Set ce to infinity.
                ceArray[i] = ce if ce is not None else np.inf
                
            # Check to see if more than 50% of EM trial were terminated or CIC moving avg is increasing
            kTooBig = np.sum(ceArray==np.inf)>(numInitParams//2)
            
            # If k is too big and k > kMin, stop increasing k and return
            if kTooBig and k > kMin:
                    break
            
            # If k == kMin
            elif kTooBig and k == kMin:
            
                # If all of the EM attempts failed, decrease kMin and try again
                if np.all(ceArray==np.inf):
                    
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
                        e = Exception('Unable to solve EM after 100 tries...aborting!')
                        raise(e)
                
                # If only some of them failed, just take what we've got and break
                else:
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
                            print('Error computing CIC during runStage!')
                            logging.error('Error while computing CIC: {}'.format(str(e)))
                        raise e
                        
                    cicList.append(cic)
                    
                    resetCounter = 0
                    
                    # Break after doing these computations
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
        
        # Verify that we have computed the CIC for each param
        try:
            assert(len(cicList)>0)
        except Exception as e:
            print('No CIC values were computed!')
            print('kMin: {}'.format(kMin))
            print('kMax: {}'.format(kMax))
            raise e
        
        try:
            assert(len(cicList)==len(bestParamsList))
        except Exception as e:
            print('Number of CIC values does not match the number of parameters!')
            raise e
        
        return bestParamsList,cicList,cicMA
    
    def run(self):
        self._results = self._run(self.initParams,self.p,self.h)
    
    def _run(self,initParams,p,h):
        """
        Run the CEM algorithm.

        Parameters
        ----------
        initParams : TYPE
            DESCRIPTION.
        p : TYPE
            DESCRIPTION.
        h : TYPE
            DESCRIPTION.

        Raises
        ------
        e
            DESCRIPTION.

        Returns
        -------
        cicArray : TYPE
            DESCRIPTION.
        paramsList : TYPE
            DESCRIPTION.
        X : TYPE
            DESCRIPTION.

        """
        
        # Reset params list
        self.paramsList = [self.initParams,]
        self.cicArray = np.zeros(shape=(self.numIters,1))
        
        self.timestamp = dttm.datetime.now().strftime('%m%d%Y_%H%M%S')
       
        # Create log file
        if self.log:
            logging.basicConfig(filename='experiment_{}.log'.format(self.timestamp),
                        level=logging.INFO)
        
        # Loop for executing the algorithm
        for s in range(self.numIters):
            
            self.s = s
            
            if self.verbose:
                print('Beginning Stage s={}'.format(s))
            
            # Determine kMin
            if s<=1:
                kMin = 1
            else:
                kMin = max(1,self.paramsList[-1].k()-3)
                # kMin = 1
            
            try:
                # Run the main operations of the stage
                bestParamsByK,cicByK,cicMA = self.runStage(s, self.paramsList[s], kMin)
            except Exception as e:
                print('Error during runStage: {}'.format(str(e)))
                if self.log:
                    print('Aborting replication {} due to error!'.format(__name__))
                    logging.error(str(e))
                
                # Write out the CEM object for diagnosing the error
                self.write()
                raise e
            
            # If this is None, we did not have enough non-zero Hx values.
            # In that case, we keep the same GMM from previous stage and continue            
            if bestParamsByK is None:
                
                if s > 0:
                    bestParamsByK,cicByK = self.bestParamsLists[-1],self.cicLists[-1]
                else:
                    bestParamsByK,cicByK = [self.initParams,],[self.cic(self.initParams),]
                
            # Record the best params and cic arrays
            self.bestParamsLists.append(bestParamsByK)
            self.cicLists.append(cicByK)
            
            # Select the best params from the stage by minimizing CIC
            bestInd = np.argmin(cicByK)
            bestParams = bestParamsByK[bestInd]
            
            # Add new params to the list
            self.paramsList.append(bestParams)
                
            self.cicArray[s] = cicByK[bestInd]
        
        return self.cicArray,self.paramsList,self.X
       
    def getResults(self):
        return self._results
    
    def writeResults(self,filename=None):
        
        if filename is None:
            filename = 'results_{}.pck'.format(self.timestamp)
        
        else:
            # Check that the file has .pck extension
            try:
                assert(filename.split(os.path.extsep)[-1]=='pck')
            except Exception as e:
                print('Use the file extension {}pck for storing results!'.format(os.path.extsep))
                raise e
        
        results = self.getResults()
        with open(filename,'wb') as f:
            pck.dump(results,f)
            
        return filename
            
    def write(self,filename=None,error=False):
    
        if filename is None:
            if error:
                filename = 'CEM_{}_ERROR.pck'.format(self.timestamp)
            else:
                filename = 'CEM_{}.pck'.format(self.timestamp)
            
        else:
            # Check that the file has .pck extension
            try:
                assert(filename.split(os.path.extsep)[-1]=='pck')
            except Exception as e:
                print('Use the file extension {}pck for storing CEM!'.format(os.path.extsep))
                raise e
        
        with open(filename,'wb') as f:
            _self = copy.deepcopy(self)
            _self.p = None
            _self.h = None
            pck.dump(_self,f)   
            
        return filename
    
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
            
            # Concatenate samples from each stage to a single numpy array
            if type(X) is list:
                _X = np.concatenate(X,axis=0)
            else:
                _X = X
                
            _alpha = np.tile(alpha,(_X.shape[0],1))
            
            # Compute density at each observation
            densities = np.zeros(shape=(_X.shape[0],params.k()))
            for j in range(params.k()):
                densities[:,j] = stat.multivariate_normal.pdf(_X,mu[j,:],sigma[j,:],allow_singular=True)
            
            densities = np.expand_dims(np.sum(np.exp(np.log(_alpha) + np.log(densities)),axis=1),axis=1)
            
            return densities
        
        return _q
    
def getAverageDensityFn(paramsList):
    
    densityFnList = []
    
    for params in paramsList:
        densityFnList.append(q(params))
        
    def _q(X):
        
        densities = [densityFn(X) for densityFn in densityFnList]
        densities = np.mean(np.stack(densities),axis=0)
        return densities
    
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
        try:
            mixtureComponents = np.random.choice(np.array(range(params.k())),size=(num,),p=alpha.squeeze().astype(np.float64))
        except Exception as e:
            print(alpha.squeeze())
            print(np.sum(alpha.squeeze()))
            raise e
    else:
        mixtureComponents = np.zeros(shape=(num,)).astype(int)

    # Generate the vectors from the mixture components
    x = np.zeros(shape=(num,mu.shape[1]))
    for i,mixtureComponent in enumerate(mixtureComponents):
        _mu = mu[mixtureComponent,:]
        _sigma = sigma[mixtureComponent,:,:]
        x[i,:] = np.random.RandomState().multivariate_normal(_mu,_sigma)
    
    return x.astype(np.longdouble)
        
        