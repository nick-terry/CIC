#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 09:40:51 2021

@author: nick
"""

import json
import numpy as np
import matplotlib.pyplot as plt

with open('replication_data.json','rb') as f:
    data = json.load(f)
    
outlier = data[70]
lik_rat = np.array(outlier['likelihood_ratios'])
lik_rat_g0 = lik_rat[lik_rat>0]
fig,ax = plt.subplots(1)
ax.violinplot([lik_rat_g0,],
                  showmeans=True,
                  quantiles=[.25,.75])

sorted_lik_rat = np.sort(lik_rat_g0)
plt.plot(sorted_lik_rat)
plt.xlabel('order statistic')
plt.ylabel('likelihood ratio')

plt.plot(sorted_lik_rat[-200:-1])
plt.xlabel('order statistic')
plt.ylabel('likelihood ratio')

# Sort rep data by rho in incr order
sort_ind = np.argsort([item['rho'] for item in data])
sorted_data = []

for i in sort_ind:
    sorted_data.append(data[i])
    
# Check the difference between nth and (n-1)th order stat likelihood ratio for each rep
delta = np.zeros(shape=(100,1))
for i in range(len(sorted_data)):
    lr = np.array(sorted_data[i]['likelihood_ratios'])
    sorted_lr = np.sort(lr,axis=0)
    delta[i] = sorted_lr[-1]-sorted_lr[-2]
    
plt.plot(delta)
plt.xlabel('order statistic (replication)')
plt.ylabel('delta')