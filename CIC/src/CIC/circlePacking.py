#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 13:39:55 2021

@author: nick
"""

import cvxpy as cvx
import numpy as np
import dccp
import matplotlib.pyplot as plt


def getPacking(n,d,r,r2=None,plot=False):

    # Adapted from https://github.com/cvxgrp/dccp/blob/master/examples/circle_packing.py    

    np.random.seed(0)
    
    if type(r) is not np.ndarray:
        r = r*np.ones((n,))
        
    if r2 is not None and type(r2) is not np.ndarray:
        r2 = r2*np.ones((n,))
    
    c = cvx.Variable((n, d))
    constr = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            constr.append(cvx.norm(cvx.vec(c[i, :] - c[j, :]), 2) >= r[i] + r[j])
            
    prob = cvx.Problem(cvx.Minimize(cvx.max(cvx.max(cvx.abs(c), axis=1) + r)), constr)
    prob.solve(method="dccp", solver="ECOS", ep=1e-2, max_slack=1e-2)
    
    if r2 is not None and plot:
        l = cvx.max(cvx.max(cvx.abs(c), axis=1) + r).value * 2
        pi = np.pi
        ratio = pi * cvx.sum(cvx.square(r)).value / cvx.square(l).value
        print("ratio =", ratio)
        
        plt.figure(figsize=(5, 5))
        circ = np.linspace(0, 2 * pi)
        x_border = [-l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_border = [-l / 2, -l / 2, l / 2, l / 2, -l / 2]
        for i in range(n):
            plt.plot(
                c[i, 0].value + r2[i] * np.cos(circ), c[i, 1].value + r2[i] * np.sin(circ), "b"
            )
        plt.plot(x_border, y_border, "g")
        plt.axes().set_aspect("equal")
        plt.xlim([-l / 2, l / 2])
        plt.ylim([-l / 2, l / 2])
        plt.show()

    return c.value

def getSigmaSq(params,X,epsilon):

    np.random.seed(0)
    
    k = params.k()
    d = params.dim()
    alpha,mu,_ = params.get()
    
    Lambda = cvx.Variable((1,))
    constr = []
    
    # Add constraints for minimum density for each mixture component/grid point combo
    for i in range(k):
        for j in range(X.shape[0]):
            constr.append(d*np.log(np.sqrt(2*np.pi)) \
                          - 2*d*cvx.log(Lambda) \
                              + .5 * Lambda**2 * cvx.norm(cvx.vec(mu[i, :] - X[j, :]), 2)**2 <= np.log(alpha[i]*k/epsilon))
    
    # Constrain Lambda to unit interval
    constr.append(Lambda >= 0)
    constr.append(Lambda <= 1)
    
    prob = cvx.Problem(cvx.Maximize(Lambda), constr)
    prob.solve(method="dccp", solver="ECOS", ep=1e-2, max_slack=1e-2)

    return 1/Lambda.value**2

if __name__ == '__main__':
    r = 3
    A = getPacking(n=5, d=2, r=r, r2=np.sqrt(r), plot=True)