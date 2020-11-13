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
        
        # Some important constants which can be changed
        # Set a threshold beyond which we consider r_x to be zero to avoid div by zero
        if 'eps' in kwargs:
            self.eps = kwargs['eps']
        else:
            self.eps = 1e-10
        
        # Set jitter used to prevent numerical issues due to zero densities
        if 'jitter' in kwargs:
            self.jitter = kwargs['jitter']
        else:
            self.jitter = 1e-300
          
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
    
    def c_bar(self,params):
        """
        Estimate the cross-entropy from the importance sampling
        distribution defined by eta, using the estimator from equation 3.7 of
        the paper.
    
        Parameters
        ----------
        params : list
            Parameters of the approximation we wish to estimate cross entropy for
    
    
        Returns
        -------
        _c_bar : float
            The estimated cross-entropy.
    
        """
        
        X = np.concatenate(self.X,axis=0)
        log_q_theta = self.log_q(params)
        llh = log_q_theta(X)
        
        _c_bar = -np.mean(llh * np.concatenate(self.Hx_WxList))
        
        return _c_bar
    
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
        
        # Compute dimension of model parameter space from the number of mixtures, k
        k = params.k()
        p = X.shape[1]
        d = (k-1)+k*(p+p*(p+1)/2)
        
        _cic = self.c_bar(params) + self.rho()*d/X.shape[0]

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
            
            densities = np.expand_dims(np.sum(_alpha*densities,axis=1),axis=1)
            
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
            for j in range(params.k()):
                log_densities[:,j] = stat.multivariate_normal.logpdf(_X,mu[j,:],sigma[j,:],allow_singular=self.allowSingular)
            
            log_densities = np.expand_dims(spc.logsumexp(np.log(_alpha)+log_densities,axis=1),axis=1)
            
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
                mixtureComponents = np.random.choice(np.array(range(params.k())),size=(num,),p=alpha.squeeze())
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
  
        # Add jitter to density to prevent div by zero error
        log_density = np.expand_dims(spc.logsumexp(log_alpha_q,axis=1),axis=1) + self.jitter
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
        covar_regularization = 10**-6 #Add ridge to covar matrix to prevent numerical instability
        
        # To make this code (slightly) more readable
        X,q = self.X,self.q
        
        gammaList = []
        # densityList = []
    
        for s in range(len(X)):
            
            x = X[s]
            gamma = self.expectation(x,q,params)
            
            gammaList.append(gamma)
        
        r_div_q_arr = np.zeros(shape=(len(X),1))
        r_div_q_gamma_arr = np.zeros(shape=(len(X),params.k()))
        mu_arr  = np.zeros(shape=(len(X),params.k(),X[0].shape[1]))
        
        for s in range(len(X)):
            
            # Compute the denominator of alpha_j expression
            r_div_q = self.Hx_WxList[s]

            # Sum over all i, store in array for stage s
            r_div_q_arr[s] = np.sum(r_div_q,axis=0)[0]
            
            # Compute the numerator for alpha_j expression/denominator for mu_j and sigma_j expressions
            r_div_q_gamma = r_div_q*gammaList[s]
            
            # Sum over all i, store in array for stage s/each j
            r_div_q_gamma_arr[s] = np.sum(r_div_q_gamma,axis=0)
            
            # Tile these quantities so that dimensions match for multiplication
            r_div_q_gamma_tile = np.repeat(np.expand_dims(r_div_q_gamma,axis=-1),X[s].shape[1],axis=-1)
            x_tile = np.repeat(np.expand_dims(X[s],axis=1),params.k(),axis=1)
            
            # Compute numerator of expression for mu_j, sum over all i, store in array for stage s/each j
            mu_arr[s] = np.sum(r_div_q_gamma_tile*x_tile,axis=0)
            
        # Compute new alpha, mu
        alpha = np.sum(r_div_q_gamma_arr,axis=0)/np.sum(r_div_q_arr)
        
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
            
        mu = np.sum(mu_arr,axis=0)/np.repeat(np.expand_dims(np.sum(r_div_q_gamma_arr,axis=0),axis=1),
                                             X[0].shape[1],axis=1)
        
        # Check that there is no inf/nan values
        try:
            assert(not np.any(np.isnan(mu)))
            assert(not np.any(mu==np.inf))
        except Exception as e:
            print('Floating point error while computing mu!')
            raise e
            
        
        sigma_arr = np.zeros(shape=(len(X),params.k(),X[0].shape[1],X[0].shape[1]))
        for s in range(len(X)):
    
            # Tile mu for each sample of stage s
            mu_tile = np.tile(mu,(X[s].shape[0],1,1))
            
            # Tile the samples of stage s for each component mean mu_j
            x_tile = np.repeat(np.expand_dims(X[s],axis=1),params.k(),axis=1)
            
            # Compute the vector X_i-mu_j
            diff = x_tile-mu_tile
            
            # Compute covar matrices w/ outer product (X_i-mu_j)(X_i-mu_j)^T
            covar = np.zeros(shape=(X[s].shape[0],params.k(),X[s].shape[1],X[s].shape[1]))
            for i in range(X[s].shape[0]):
                for j in range(params.k()):
                    covarMat = np.outer(diff[i,j,:],diff[i,j,:])
                    covar[i,j,:,:] = covarMat
            
            # Compute the denominator of expression for sigma_j
            r_div_q = self.Hx_WxList[s]
            r_div_q_gamma = r_div_q*gammaList[s]
            
            # Tile the denominator 
            r_div_q_gamma_tile = np.tile(np.expand_dims(r_div_q_gamma,axis=(-1,-2)),(1,1,X[0].shape[1],X[0].shape[1]))
        
            sigma_arr[s,:,:,:] = np.sum(r_div_q_gamma_tile*covar,axis=0)
            
        regularization_diag = covar_regularization * np.repeat(np.expand_dims(np.eye(X[0].shape[1]),
                                                                             axis=0),params.k(),axis=0)
        
        # Compute new sigma
        sigma = np.sum(sigma_arr,axis=0)/\
            np.tile(np.expand_dims(np.sum(r_div_q_gamma_arr,axis=0),axis=(-1,-2)),
                    (1,1,X[0].shape[1],X[0].shape[1])).squeeze()\
                    + regularization_diag #Add ridge to prevent numerical instability
         
        # Check that there is no inf/nan values
        try:
            assert(not np.any(np.isnan(sigma)))
            assert(not np.any(sigma==np.inf))
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
            condNum = np.linalg.cond(sigma)
            if np.any(condNum>condThresh):
                
                # If covar is not well conditioned (i.e. model is overfit), abort the EM procedure
                if not retCE:
                    return None
                else:
                    errStr = ''
                    if params.k()==1:
                        numEventsOcc = np.zeros(self.X.shape[0])
                        for i in range(self.X.shape[0]):
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
                    chosenInd.append(zeroInd[choice])
                    
                chosenInd = np.array(chosenInd).astype(int)
                    
                # Create XBar matrix by concatenating the data vectors
                # Get the sampled data
                XBar = X[chosenInd,:]
                
                # Compute covar matrix for GMM params
                p = X.shape[-1]
                covar = np.cov(XBar,rowvar=False)
                tr = np.trace(covar) if len(covar.shape)>0 else covar
                covar = 3/p*tr*np.eye(p)
                covar = np.tile(np.expand_dims(covar,axis=0),(k,1,1))
                
                # Create GMMParams object and add to list
                initParams  = GMMParams(np.ones(k)/k, XBar, covar, p)
                initParamsList.append(initParams)
            
        return initParamsList
     
    
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
            q_theta = self.q(params)
            px = self.p(x)
            Wx = px/q_theta(x)
            
            r_x = Hx * px
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
            
            if self.verbose:
                print('Beginning Stage s={}'.format(s))
            
            # Determine kMin
            if s<=1:
                kMin = 1
            else:
                kMin = max(1,self.paramsList[-1].k()-3)
            
            try:
                # Run the main operations of the stage
                bestParamsByK,cicByK,cicMA = self.runStage(s, self.paramsList[s], kMin)
            except Exception as e:
                print('Error during runStage: {}'.format(str(e)))
                if self.log:
                    print('Aborting replication {} due to error!'.format(__name__))
                    logging.ERROR(str(e))
                # Write out the CEM object for diagnosing the error
                self.write()
                raise e
            
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