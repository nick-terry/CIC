#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:22:14 2020

@author: nick
"""

import matlab
import matlab.engine
import io
import numpy as np
import scipy.stats as stat

class SimulationEngine:
    
    def __init__(self,engine=None,seed=42,simulation='39-bus',matlabPrint=False):
        
        #Start a MATLAB engine for running COSMIC simulation
        if engine is None:
            self.eng = matlab.engine.start_matlab()
        else:
            self.eng = engine
        
        
        self.matlabPrint = matlabPrint
        if not matlabPrint:
            self.stdout = io.StringIO()

        #Set random seed
        self.seed = seed
        self.rs = np.random.RandomState(self.seed)
        
        #Define the simulation to run
        self.simFns = {'9-bus': self.eng.sim_case9_fn,
                      '39-bus': self.eng.sim_case39_fn,
                      '2383-bus': self.eng.sim_case2383_fn}
        self.simToNumBranches = {'9-bus':9,
                                 '39-bus':46,
                                 '2383-bus':2896}
        
        try:
            assert(simulation in self.simFns.keys())
        except Exception as e:
            print('{} is not a valid COSMIC simulation config!')
            raise(e)
        
        self.simulation = simulation
        self.numBranches = self.simToNumBranches[self.simulation]
        self.simFn = self.simFns[self.simulation]
    
    def simulate(self,params):
        """
        Run a stochastic simulation of an N-2 contingency using COSMIC.

        Parameters
        ----------
        params : numpy array
            The failure rates of each branch of the network. These are used
            as the parameters of the distribution which gives the
            time to failure of each branch.

        Raises
        ------
        e
            DESCRIPTION.

        Returns
        -------
        br1 : int
            The first failed branch.
        br2 : TYPE
            The second failed branch.
        blackout : boolean
            Whether or not a blackout occurred.
        lostDemand : TYPE
            Amount of demand lost due to outage.

        """
        
        #Check that the right number of failure rate params are passed
        try:
            assert(params.shape==(self.numBranches,1))
        except Exception as e:
            print('Incorrect number of failure rate params!')
            raise e
        
        #Generate time to failure of each branch
        timeToFailure = np.array([self.rs.normal(loc=x,scale=3) for x in params])
        #Get the two branches which failed
        sortedInd = np.argsort(timeToFailure,axis=0)[:2]
        br1,br2 = int(sortedInd[0]),int(sortedInd[1])
        
        #Run COSMIC simulation with N-2 contingency of br1,br2
        blackout,lostDemand = self._simulate(br1,br2)
        
        return br1,br2,blackout,lostDemand    
        
    def _simulate(self,br1,br2,verbose=False):
        """
        Run COSMIC simulation of N-2 contingency in MATLAB

        Parameters
        ----------
        br1 : int
            First branch failure.
        br2 : int
            Second branch failure.

        Returns
        -------
        blackout : boolean
            Whether or not a blackout occurred.
        lostDemand : float
            Amount of demand lost due to outage.

        """
        
        # Need to specify nargout=2 to get two results correctly
        try:
            if self.matlabPrint:
                blackout,lostDemand = self.simFn(br1+1,br2+1,verbose, nargout=2)
            else:
                blackout,lostDemand = self.simFn(br1+1,br2+1,verbose, nargout=2, stdout=self.stdout)
        # If the simulation does not converge, declare a blackout
        except matlab.engine.MatlabExecutionError:
            blackout,lostDemand = 1,0
        return blackout,lostDemand

if __name__=='__main__':    
    
    # Run a test of the SimulationEngine
    sim = SimulationEngine(matlabPrint=False)
    
    # Model failure time as N(10,3).
    rates = 10 * np.ones(shape=(sim.numBranches,1))
    #result = sim.simulate(rates)
    result = sim._simulate(37,13)
