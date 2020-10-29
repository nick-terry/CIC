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
import scipy.stats as stat
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
        h : TYPE
            Simulator evaluation
        **kwargs : dict
            Keyword arguments.

        """
        
        self.initParams = initParams
        self.dim = initParams.get()[1].shape[1]
        self.p = p
        self.h = h
        
        self.paramsList = []
        
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
            self.jitter = 1e-100
          
        if 'numIters' in kwargs:
            self.numIters = kwargs['numIters']
        else:
            self.numIters = 10
        
        if 'sampleSize' in kwargs:
            self.sampleSize = kwargs['sampleSize']
        else:
            self.sampleSize = 1000
    
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
    
    def entropy(self,x,r_x,q,params,newParams):
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
    
    def c_bar(self,X,rList,q,paramsList,newParams):
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
            
            cumulative_c_bar += np.sum(self.entropy(x,r_x,q,oldParams,newParams))
        
        _c_bar = -1/X.shape[0]/X.shape[1] * cumulative_c_bar
        return _c_bar
    
    def rho(self,X,rList,q,paramsList):
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
    
    def cic(self,X,rList,q,paramsList):
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
        
        _cic = self.c_bar(X,rList,q,oldParamsList,newParams)\
            + self.rho(X,rList,q,paramsList)*d/X.shape[0]/X.shape[1]
        
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
                
                density = np.expand_dims(np.sum(_alpha*densities,axis=1),axis=1) + self.jitter
                
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
                    
                    densityList.append(np.expand_dims(np.sum(_alpha*densities,axis=1),axis=1) + self.jitter)
                
                return np.concatenate(densityList)
        
        return _q
    
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
            mixtureComponents = np.random.choice(np.array(range(params.k())),size=(num,),p=alpha.squeeze())
        else:
            mixtureComponents = np.zeros(shape=(num,)).astype(int)
    
        # Generate the vectors from the mixture components
        x = np.zeros(shape=(num,self.dim))
        for i,mixtureComponent in enumerate(mixtureComponents):
            _mu = mu[mixtureComponent,:]
            _sigma = sigma[mixtureComponent,:,:]
            x[i,:] = np.random.RandomState().multivariate_normal(_mu,_sigma)
        
        return x
    
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
        density = np.expand_dims(np.sum(alpha_q,axis=1),axis=1) + self.jitter
        _density = np.tile(density,(1,params.k()))
        
        gamma = alpha_q/_density
        
        return gamma,density
    
    def emIteration(self,X,r_xList,q):
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
            gamma,density = self.expectation(x,q,self.paramsList[s])
            
            gammaList.append(gamma)
            densityList.append(density)
    
        params = self.paramsList[-1]
        
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
            # Tile the denominator 
            # TODO: fix error here. Not sure why this fails. Should accept tuples
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
    
    def updateParams(self,X,rList,q,eps=1e-2,maxiter=10,ntrials=1,retCE=False):
        """
        Update params by performing EM iterations until converged
    
        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        rList : TYPE
            DESCRIPTION.
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
        _params = self.paramsList[-1]
        paramTrials = [_params.getCopy() for trial in range(ntrials)]
        ceArray = np.zeros(shape=(ntrials,1))
        for trial in range(ntrials):
            
            params = paramTrials[trial]
            i = 0
            delta = np.inf
            
            ce = self.c_bar(X,rList,q,self.paramsList,params)
            
            # Loop until EM converges
            while delta >= eps*np.abs(ce) and i < maxiter:
                
                #print('Old Mu: {}'.format(params.get()[1]))
                
                alpha,mu,sigma = self.emIteration(X, rList, q)
                
                #print('New Mu: {}'.format(mu))
                
                params.update(alpha,mu,sigma)
                self.paramsList[-1] = params
                
                _ce = self.c_bar(X,rList,q,self.paramsList,params)
                
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
        self.paramsList[-1] = _params
        
        if not retCE:
            return bestParams
        else:
            return bestParams,ceArray[bestParamsInd]
    
    def run(self):
        self._results = self._run(self.initParams,self.p,self.h)
        
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
            
    def write(self,filename=None):
    
        if filename is None:
            filename = 'CEM_{}.pck'.format(self.timestamp)
        
        else:
            # Check that the file has .pck extension
            try:
                assert(filename.split(os.path.extsep)[-1]=='pck')
            except Exception as e:
                print('Use the file extension {}pck for storing CEM!'.format(os.path.extsep))
                raise e
        
        with open(filename,'wb') as f:
            pck.dump(self,f)   
            
        return filename
    
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
    
        numIters = self.numIters
        sampleSize = self.sampleSize
        paramSize = initParams.get()[1].shape[1]
        
        # Reset params list
        self.paramsList = [self.initParams,]
        rList = []
        cicArray = np.zeros(shape=(numIters,1))
        
        # Create log file
        self.timestamp = dttm.datetime.now().strftime('%m%d%Y_%H%M%S')
        logging.basicConfig(filename='experiment_{}.log'.format(self.timestamp),
                        level=logging.INFO)
        
        logging.info('Initial GMM Params:\n'+str(initParams))
        
        # Loop for executing the algorithm
        for i in range(numIters):
            
            print('Beginning Stage s={}'.format(i))
            
            # If first stage, need to choose best GMM from init list
            if i==0:
            
                #x = np.random.uniform(-5,5,size=(sampleSize,self.dim))
                x = self.generateX(self.paramsList[-1],sampleSize)
                
                X = np.expand_dims(x,axis=0)
                
            else:
               
                x = self.generateX(self.paramsList[-1],sampleSize)

                # Update the parameters of the GMM
                _x = np.expand_dims(x,axis=0)
                
                X = np.concatenate([X,_x],axis=0)
            
            # Run simulation and compute likelihood
            try:
                r_x = self.r(x)
            except Exception as e:
                logging.error('Error during r(x): {}'.format(str(e)))
                raise e
                
            rList.append(r_x)
            
            try:
                params = self.updateParams(X, rList, self.q, ntrials=10)
            except Exception as e:
                logging.error('Error while updating params: {}'.format(str(e)))
                raise e
                
            # Add new params to the list
            self.paramsList.append(params)
        
            # Compute the CIC
            try:
                cic = self.cic(X,rList,self.q,self.paramsList)
            except Exception as e:
                logging.error('Error while computing CIC: {}'.format(str(e)))
                raise e
                
            cicArray[i] = cic
            logging.info('CIC: {}'.format(cic))    
    
        return cicArray,self.paramsList,X