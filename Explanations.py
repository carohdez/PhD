# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:05:58 2019

@author: Carolina
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import xml.etree.ElementTree
import math
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn import metrics
import seaborn as sns
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp
# =============================================================================
# extraction of data 
# =============================================================================
# Data
#data = pd.read_csv('data_explanations_2019-09-20_07-45.csv')
##workers = pd.read_csv('workers.csv')
data = pd.read_csv('data4.csv')
data_original=data

#===================================================================
# Load data
#===================================================================

# Highly restrictive (151)
df = pd.read_csv('data_filtered.csv')

# Final (212): Using final cases documented in Google drive 
finalCases = pd.read_csv('FinalCases.csv')
finalCases = finalCases[finalCases.INCLUDE==1]
casesIds=np.array(finalCases.CASE)
casesIds=casesIds.astype(str)
df=data[data_original.CASE.isin(casesIds)]


df.to_csv('data_final.csv')

# count of cases and conditions
df.RG01.value_counts()
np.sum(df.RG01.value_counts())

# drop validation questions
df=df.drop(['UT02_16','UT02_17','EE02_20', 'EE02_21'], axis=1)

df['UT_rational'] = df.loc[:, 'UT02_01':'UT02_05'].mean(axis=1)
df['UT_intuitive'] = df.loc[:, 'UT02_06':'UT02_10'].mean(axis=1)
df['UT_rational_median'] = df.loc[:, 'UT02_01':'UT02_05'].median(axis=1)
df['UT_intuitive_median'] = df.loc[:, 'UT02_06':'UT02_10'].median(axis=1)
df['UT_intuitive_round'] = round(df.loc[:, 'UT02_06':'UT02_10'].mean(axis=1))
df['UT_so_awareness'] = df.loc[:, 'UT02_11':'UT02_15'].mean(axis=1)
df['UT_so_awareness_median'] = df.loc[:, 'UT02_11':'UT02_15'].median(axis=1)
#df['explanation_quality'] = df.loc[:, 'EE02_01':'EE02_05'].mean(axis=1) 
df['explanation_quality'] = df.loc[:, 'EE02_01':'EE02_02'].mean(axis=1) 
df['transparency'] = df.loc[:, 'EE02_04':'EE02_05'].mean(axis=1)
df['effectiveness'] = df.loc[:, 'EE02_06':'EE02_08'].mean(axis=1)
df['efficiency'] = df.loc[:, 'EE02_09']
df['trust'] = df.loc[:, 'EE02_10':'EE02_16'].mean(axis=1)
df['social_presence'] = df.loc[:, 'EE02_17':'EE02_19'].mean(axis=1)

# users with level L,non intuitive but with high explanation quality
df.DI01_01.mean()
df.DI02.value_counts() #1: male, 2 female
def get_level(condition):
    if (condition==1) | (condition==3) | (condition==5): return 'L'
    elif (condition==2) | (condition==4) | (condition==6): return 'H'

def get_type(condition):
    if (condition==1) | (condition==2): return 'A'
    elif (condition==3) | (condition==4): return 'S'
    elif (condition==5) | (condition==6): return 'R'
def get_generation(age):
    if (age<20): return 'Z'
    elif (age>=20) & (age<40): return 'Y'
    elif (age>=40) & (age<55): return 'X'
    elif (age>=55) & (age<73): return 'BB'
    elif (age>=73) : return 'G'
def sig(p):
    if p<0.001: return '***'
    elif p<0.01: return '**'
    elif p<0.05: return '*'
    return ''   
def get_int_id(case, col1, col2):
    return df[df['CASE']==case][col1].values[0]+ '-' + str(int(df[df['CASE']==case][col2].values[0]))
def get_int_id_bin(case, col1, col2):
    return df[df['CASE']==case][col1].values[0]+ '-' + df[df['CASE']==case][col2].values[0]
def get_DMS(rating):
    if (rating==1) | (rating==2) | (rating==3): return 'L_DMS'
    elif (rating==4) | (rating==5): return 'H_DMS'
df['level']=df.RG01.apply(lambda x: get_level(x))
df['type']=df.RG01.apply(lambda x: get_type(x))
df['generation']=df.DI01_01.apply(lambda x: get_generation(x))

# =============================================================================
# Significant tests 2way ANOVA
# =============================================================================
#model = ols('explanation_quality ~ level*UT_rational + type*UT_rational', df).fit()

df['explanation_quality'] = df.loc[:, 'EE02_01':'EE02_02'].mean(axis=1) 

#dms_vble=['UT02_02']
dms_vbles=['UT02_'+ ('0'+str(x) if x<10 else str(x)) for x in range(1, 16, 1)] # individual variables
dms_vbles=['UT_rational','UT_intuitive','UT_so_awareness'] # aggregated constructs
#dms_vble=['social_awareness']
#dms_vbles=['DI02','DI03','generation'] # dempgraphics
score_vbles=['explanation_quality','transparency','EE02_04','effectiveness','efficiency','trust','social_presence']
score_vbles=['effectiveness']
#score_vbles=['EE02_04','EE02_07','EE02_11','EE02_13','EE02_15']


for score_vble in score_vbles:
    print("Score: ", score_vble, "------------------------------------------------------")
    for dms_vble in dms_vbles:
        model = ols(score_vble+' ~ level*'+dms_vble+' + type*'+dms_vble, df).fit() # DMS
        if model.f_pvalue <  0.05: # only print significant models
            print("Significant model dms_vble: ", dms_vble, "-------------")
            print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")
            model.summary()
            res = sm.stats.anova_lm(model, typ= 2)
            res['sig']=res['PR(>F)'].apply(lambda x: sig(x))
            print(res)

# post-hoc test factors ------------------------------------------------
mc = statsmodels.stats.multicomp.MultiComparison(df['explanation_quality'] , df['type'])
mc_results = mc.tukeyhsd(alpha=0.05)
print(mc_results)

# =============================================================================
# Plots
# =============================================================================
groups=['H','L']
condition='level'
# Plot Type
groups=['A','S','R']
condition='type'
covariates=['UT_so_awareness']
covariates=['UT_intuitive','UT_so_awareness']

# final paper graph
groups=['H','L']
condition='level'
xlabel='Intuitive decision-making score'
covariates=['UT_intuitive']
#xlabel='Social awareness score'
ylabel='Mean of explanation quality'
f_size=14
t_size=8
l_size=25

colors=['magenta', 'turquoise']
clr=0
for x_vble in covariates:
    fig, ax = plt.subplots()
    for i in groups:
        if i=='all':
            X = df[x_vble].values.reshape(-1, 1)  
            y = df[y_vble].values.reshape(-1, 1) 
        else:
            X = df[df[condition]==i][x_vble].values.reshape(-1, 1)  
            y = df[df[condition]==i][y_vble].values.reshape(-1, 1) 
        linear_regressor = LinearRegression() 
        linear_regressor.fit(X, y)
        y_pred = linear_regressor.predict(X) 
        #ax.scatter(X, y)
        ax.plot(X, y_pred, label=i, linewidth=2, color=colors[clr])
        clr=clr+1
    leg=ax.legend()    
    ax.set_xlabel(xlabel, fontsize=f_size)
    ax.set_ylabel(ylabel, fontsize=f_size)
    ax.tick_params(axis='both', which='major', labelsize=t_size)
    #ax.set_xlim(2,5)
    ax.set_ylim(2.5,5)
    l=plt.legend()
    l.fontsize=l_size
    l.get_texts()[0].set_text('High level')
    l.get_texts()[1].set_text('Low level')
    plt.savefig('level.png', dpi=600)

groups=['A','S','R']
condition='type'
covariates=['UT_so_awareness'] 
xlabel='Social awareness score'    
for x_vble in covariates:
    fig, ax = plt.subplots()
    for i in groups:
        if i=='all':
            X = df[x_vble].values.reshape(-1, 1)  
            y = df[y_vble].values.reshape(-1, 1) 
        else:
            X = df[df[condition]==i][x_vble].values.reshape(-1, 1)  
            y = df[df[condition]==i][y_vble].values.reshape(-1, 1) 
        linear_regressor = LinearRegression() 
        linear_regressor.fit(X, y)
        y_pred = linear_regressor.predict(X) 
        #ax.scatter(X, y)
        ax.plot(X, y_pred, label=i, linewidth=2)
    leg=ax.legend()    
    ax.set_xlabel(xlabel, fontsize=f_size)
    ax.set_ylabel(ylabel, fontsize=f_size)
    ax.tick_params(axis='both', which='major', labelsize=t_size)
    #ax.set_xlim(2,5)
    ax.set_ylim(2.5,5)
    l=plt.legend()
    l.fontsize=l_size
    #l.get_texts()[0].set_text('High level')
    #l.get_texts()[1].set_text('Low level')
    l.get_texts()[0].set_text('Aggregation')
    l.get_texts()[1].set_text('Summary')
    l.get_texts()[2].set_text('Review')
    plt.savefig('type.png', dpi=600)
    
    
#x_vble='UT_rational'#'UT_intuitive'
y_vble='explanation_quality'#'social_presence'

for x_vble in covariates:
    fig, ax = plt.subplots()
    for i in groups:
        if i=='all':
            X = df[x_vble].values.reshape(-1, 1)  
            y = df[y_vble].values.reshape(-1, 1) 
        else:
            X = df[df[condition]==i][x_vble].values.reshape(-1, 1)  
            y = df[df[condition]==i][y_vble].values.reshape(-1, 1) 
        linear_regressor = LinearRegression() 
        linear_regressor.fit(X, y)
        y_pred = linear_regressor.predict(X) 
        ax.scatter(X, y)
        ax.plot(X, y_pred, label=i)
    leg=ax.legend()    
    ax.set_xlabel(x_vble)
    ax.set_ylabel(y_vble)
    #ax.set_xlim(2,5)
    ax.set_ylim(2,5)
linear_regressor.score(X, y)
c=np.corrcoef(X, y)
# median graphs--------------------------
groups=['H','L']
condition='level'
x_vble='UT_intuitive_median'#'UT_intuitive_round'#'UT_intuitive_median'
y_vble='explanation_quality'#'social_presence'
fig, ax = plt.subplots()
for i in groups:
    X = df[df[condition]==i].groupby([x_vble])[y_vble].mean().index.values 
    y = df[df[condition]==i].groupby([x_vble])[y_vble].mean()
    ax.plot(X, y, label=i)
leg=ax.legend()    
ax.set_xlabel(x_vble)
ax.set_ylabel('Mean:'+y_vble)
ax.set_xlim(1,5.1)
ax.set_ylim(1,5.1)

aux=df[(df.UT02_07==1)][['explanation_quality','level','type','UT02_08']]
#-------------- linear tendency
df[['UT_so_awareness_median', 'trust']].group_by('UT_so_awareness').plot()
df.groupby(['UT_so_awareness_median'])['trust'].mean().plot()
df.groupby(['UT_so_awareness'])
df.UT_so_awareness_median.value_counts()

X = df['UT_so_awareness'].values.reshape(-1, 1)  
y = df['effectiveness'].values.reshape(-1, 1) 
            
linear_regressor = LinearRegression() 
linear_regressor.fit(X, y)
y_pred = linear_regressor.predict(X) 

fig, ax = plt.subplots()
y_pred = linear_regressor.predict(X) 
#ax.scatter(X, y)
ax.plot(X, y_pred)
leg=ax.legend()    
ax.set_xlabel('UT_so_awareness')
ax.set_ylabel('effectiveness')


# Decision style population-------------------
plt.figure()
df[['UT_intuitive','UT_rational','UT_so_awareness']].plot.hist(bins=5, ylim=(0,80))

plt.figure()
df[['UT_intuitive']].plot.hist(bins=5, ylim=(0,80))
plt.grid(axis='y', alpha=0.75)
df[['UT_rational']].plot.hist(bins=5, ylim=(0,80))
plt.grid(axis='y', alpha=0.75)
df[['UT_so_awareness']].plot.hist(bins=5, ylim=(0,80))
plt.grid(axis='y', alpha=0.75)


#correlation matrix, showing correlation between each variable and all the others -------------------
import seaborn as sns
df['UT_intuitive_but8'] = df[['UT02_06','UT02_07','UT02_09','UT02_10']].mean(axis=1)
['UT02_'+ ('0'+str(x) if x<10 else str(x)) for x in range(6, 11, 1)]
X=df[['UT02_06', 'UT02_07', 'UT02_08', 'UT02_09', 'UT02_10', 'UT_intuitive', 'UT_intuitive_but8']]
Xcorr = X.corr() 
X.corr().head()
sns.heatmap(Xcorr, cmap = 'bwr')

# cluster intuitive - rational
X = df['UT_intuitive'].values.reshape(-1, 1)  
y = df['UT_rational'].values.reshape(-1, 1) 
fig, ax = plt.subplots()
ax.scatter(X, y)
leg=ax.legend()    
ax.set_xlabel('UT_intuitive')
ax.set_ylabel('UT_rational')




X = df['UT_rational'].values.reshape(-1, 1)  
y = df['UT_intuitive'].values.reshape(-1, 1) 
fig, ax = plt.subplots()
ax.scatter(X, y)
leg=ax.legend()    
ax.set_xlabel('UT_rational')
ax.set_ylabel('UT_intuitive')



df[['UT_rational','UT_intuitive', 'UT_so_awareness']].plot.scatter(x='UT_rational',y='UT_intuitive', c='UT_so_awareness', colormap='viridis')






# =============================================================================
# Additional Significant tests 2way ANOVA
# =============================================================================
#model = ols('explanation_quality ~ level*UT_rational + type*UT_rational', df).fit()

df['explanation_quality'] = df.loc[:, 'EE02_01':'EE02_02'].mean(axis=1) 


dms_vbles=['UT02_'+ ('0'+str(x) if x<10 else str(x)) for x in range(1, 16, 1)] # individual variables
dms_vbles=['UT_rational','UT_intuitive','UT_so_awareness'] # aggregated constructs
dms_vbles=['UT_rational','UT_intuitive'] # aggregated constructs
dms_vbles=['UT_so_awareness']
#dms_vbles=['DI02','DI03','generation'] # dempgraphics
score_vbles=['explanation_quality','transparency','effectiveness','efficiency','trust','social_presence']
score_vbles=['effectiveness']
#score_vbles=['EE02_04','EE02_07','EE02_11','EE02_13','EE02_15']
score_vble='transparency'
#df['UT_intuitive'] = df.loc[:, 'UT02_09':'UT02_10'].mean(axis=1)

df['UT_intuitive'] = df[['UT02_07','UT02_09','UT02_08','UT02_10']].mean(axis=1)
,'UT02_01','UT02_02','UT02_04','UT02_05'
df['UT_rational'] = df[['UT02_03','UT02_04']].mean(axis=1)
df['UT_intuitive'] = df[['UT02_06','UT02_07','UT02_09','UT02_10']].mean(axis=1)
for score_vble in score_vbles:
    print("Score: ", score_vble, "------------------------------------------------------")
    for dms_vble in dms_vbles:
        model = ols(score_vble+' ~ level*'+dms_vble+' + type*'+dms_vble, df).fit() # DMS
        #model = ols(score_vble+' ~ level*type', df).fit() # DMS
        if model.f_pvalue <  0.05: # only print significant models
            print("Significant model dms_vble: ", dms_vble, "-------------")
            print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")
            model.summary()
            res = sm.stats.anova_lm(model, typ= 2)
            res['sig']=res['PR(>F)'].apply(lambda x: sig(x))
            print(res)


df['UT_intuitive'] = df[['UT02_06','UT02_07']].mean(axis=1)
aux=df[['UT02_08','UT_intuitive']]

# post-hoc test factors ------------------------------------------------
mc = statsmodels.stats.multicomp.MultiComparison(df['EE02_04'] , df['type'])
mc_results = mc.tukeyhsd(alpha=0.01)
#mc_results = mc.tukeyhsd()
print(mc_results)




df[['UT_intuitive', 'UT02_08', 'UT_intuitive'-'UT02_08' ]] 
df1=df['UT_intuitive'] - df['UT02_08'] 
aux=df[['CASE','UT_intuitive', 'UT02_08']] [abs(df1)>0.75]
aux=pd.Series(df[['CASE']] [abs(df1)>1])

aux=[]

# single model, no DMS ------------------------------------------------
#model = ols(score_vble+' ~ level*'+dms_vble+' + type*'+dms_vble, df).fit() # DMS
score_vble='efficiency'
model = ols(score_vble+' ~ level*type', df).fit() # no DMS
#if model.f_pvalue <  0.05: # only print significant models
print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")
model.summary()
res = sm.stats.anova_lm(model, typ= 2)
res['sig']=res['PR(>F)'].apply(lambda x: sig(x))
print(res)

# post-hoc test factors ------------------------------------------------
mc = statsmodels.stats.multicomp.MultiComparison(df['efficiency'] , df['type'])
mc_results = mc.tukeyhsd(alpha=0.05)
print(mc_results)