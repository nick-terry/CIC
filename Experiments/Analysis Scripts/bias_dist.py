#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 09:47:21 2020

@author: nick
"""

import matplotlib.pyplot as plt
import pandas as pd

# nojitter = pd.read_csv('bias_results_nojitter.csv')
# wjitter = pd.read_csv('bias_results.csv')

# labels = ['With Jitter','Without Jitter']
# fig,ax = plt.subplots(1)
# ax.violinplot([wjitter['rho'].values,nojitter['rho'].values],
#               showmeans=True,
#               quantiles=[[.25,.75],]*2)
# ax.set_xticks(range(1, len(labels) + 1))
# ax.set_xticklabels(labels)
# ax.set_ylabel('Estimated event probability')

# filenames = ['results_10_diag.csv','results_5_diagonal.csv','results_3_diag.csv']
filenames = ['bias_results_p2.csv',]*2
labels = ['Bias Test',]*2
# truePVals = [.02345,.006258,.006336]
# truePVals = [1.177475e-05,]*2
# truePVals = [0.00816563,]*2
# truePVals = [0.03084614,]*2
truePVals = [0.04875993,]*2

fig,axes = plt.subplots(1,len(filenames))
for ax,filename,label,trueP in zip(axes,filenames,labels,truePVals):
    
    data = pd.read_csv(filename)
    ax.violinplot([data['rho'].values,],
                  showmeans=True,
                  quantiles=[.25,.75])
    ax.axhline(trueP,color='orange',label='True Event Probability')
    ax.set_xticks([1,])
    ax.set_xticklabels([label,])
    ax.set_ylabel('Probability')
    ax.legend()