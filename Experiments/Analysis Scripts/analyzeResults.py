#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 09:06:56 2021

@author: nick
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import scipy.stats as stat

def rmse( g ):
    rmse = np.sqrt( mean_squared_error( g['TrueProb'], g['rho'] ) )
    return pd.Series( dict( rmse = rmse ) )

fnameStr = 'results_p{}_unif.csv'
dfList = []

# True failure probabilities as estimated using CMC w/ 100,000 samples
# p=2 : 0.006326
# p=3 : 0.006366
# p=4 : 0.006257

pVals = (2,3,4,5,10,20,42)
trueProbs = (0.006326,
             0.006366,
             0.006257,
             0.0165,
             0.00163,
             0.0015,
             0.00147)
trueProbsDict = dict(zip(pVals,trueProbs))

nSamples = 20000

for p,trueProb in zip(pVals,trueProbs):

    df = pd.read_csv(fnameStr.format(p))
    dfList.append(df)
    
    fig,ax = plt.subplots(1)
    ax.violinplot(df['rho'].values,
                  showmeans=True,
                  quantiles=[.25,.75])
    ax.axhline(trueProb,color='orange',label='True Event Probability')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.set_title('p={}'.format(p))

nReps = len(df)/len(pVals)

df = pd.concat(dfList)
tpDF = pd.DataFrame([pVals,trueProbs]).T
tpDF.columns = ['p','TrueProb']
tpDF['p'] = tpDF['p'].astype(np.int64)
tpDF = tpDF.set_index('p')
df['TrueProb'] = tpDF.loc[df['p'].values].values

summaryDF = df.groupby('p')['rho'].agg(Mean='mean',Std='std')
summaryDF['RMSE'] = df.groupby('p').apply(rmse)
summaryDF['TrueProb'] = trueProbs
summaryDF['rmseOverTrueProb'] = summaryDF['RMSE']/summaryDF['TrueProb']
# summaryDF['StdErr'] = summaryDF['Std']/np.sqrt(nReps)
summaryDF['StdErr'] = summaryDF['Std']
summaryDF['nCMC'] = summaryDF['Mean']*(1-summaryDF['Mean'])/summaryDF['StdErr']**2
summaryDF['cmcRatio'] = nSamples/summaryDF['nCMC'] 

# Add the mean over all replicates to OG dataframe
df['pMean'] = summaryDF.loc[df['p']].Mean.values

# Plot RMSE/rho vs. p
fig,ax = plt.subplots(1)
p = summaryDF.index
y = summaryDF['rmseOverTrueProb']
ax.plot(p,y)
ax.set_xlabel('Dimension (p)')
ax.set_ylabel(r'RMSE/$\rho$')

# Plot RMSE/rho vs. p
fig,ax = plt.subplots(1)
p = summaryDF.index
cv = summaryDF['Std']/summaryDF['Mean']
cihw = stat.t.ppf(q=.95,df=100-1) * cv/np.sqrt(2*100)
ax.errorbar(p,cv,cihw)
ax.set_xlabel('Dimension (p)')
ax.set_ylabel(r'Coefficient of Variation')

# Plot RMSE/rho vs. p
# fig,ax = plt.subplots(1)
# p = summaryDF.index
# y = summaryDF['rmseOverTrueProb']
# ax.plot(p,y)
# ax.set_xlabel('Dimension (p)')
# ax.set_ylabel(r'RMSE/$\rho$')

minMax = df.groupby('p')['rho'].agg(Max='max',Min='min')
