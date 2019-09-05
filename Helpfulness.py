# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:59:27 2019

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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve


# =============================================================================
# extraction of data 
# =============================================================================

# Extract helpfulness and reasons for all reviews
codes = pd.read_csv('QuestionCodes.csv')
workers = pd.read_csv('workers.csv')
validation_q = pd.read_csv('validationQuestions.csv')
#data = pd.read_csv('data_helpfulness_2019-03-14_10-28.csv')

# Batches up to 5 
data = pd.read_csv('data_helpfulness_2019-03-20_16-45.csv')
#data = pd.read_csv('data_helpfulness_2019-03-13_14-10.csv')
#data = pd.read_csv('Helpfulness_Data - TMP2.csv')

# Responses for the validation questions
responses_vq = np.array([['VQ04',	4],['VQ05',	2],['VQ06',	3],['VQ07',	3],['VQ08',	3],['VQ09',	2]])
# Codes pages reviews
pages_reviews = np.array(['08','09','10','11','12','13','16','17','18','19','20','21','24','25','26','27','28','29'])

# Extract only workers who finsih the survey

data=data[(data['FINISHED']=='1') | ((data['FINISHED']=='0') & (pd.notna(data['TIME030'])))]

data['correct_responses'] = pd.Series(np.zeros(len(data['FINISHED'])), index=data.index)
data['av_time_review'] = pd.Series(np.zeros(len(data['FINISHED'])), index=data.index)
data['rex_vq_1'] = data['rex_vq_2'] = data['castle_vq_1'] = data['castle_vq_2'] = data['omni_vq_1'] = data['omni_vq_2'] = pd.Series(np.zeros(len(data['FINISHED'])), index=data.index)

#Get list of workers who finished
str_workers=''
for i,row in data.iterrows():
    str_workers=str_workers+' '+row['WI01_01']

#print(str_workers)

# Get the nuumber of correct responses to validation questions
def correct_responses(response,id_question):
    if ((((id_question==4) & (response == '4')) | ((id_question==5) & (response == '2'))) | (((id_question==5) & (response == '2'))) | (((id_question==6) & (response == '3'))) | (((id_question==7) & (response == '3')))  | (((id_question==8) & (response == '3'))) | (((id_question==9) & (response == '2'))) ):
        return 1 
    else: return 0 
   
data['correct_responses']=data.VQ04.apply(lambda x: correct_responses(x,4))+data.VQ05.apply(lambda x: correct_responses(x,5))+data.VQ06.apply(lambda x: correct_responses(x,6))+data.VQ07.apply(lambda x: correct_responses(x,7))+data.VQ08.apply(lambda x: correct_responses(x,8))+data.VQ09.apply(lambda x: correct_responses(x,9))

# Get the average time per review---------------

def av_time_review(l):
    return (pd.to_numeric(l[0])+pd.to_numeric(l[1]) +pd.to_numeric(l[2]) +pd.to_numeric(l[3]) +pd.to_numeric(l[4]) +pd.to_numeric(l[5]) +pd.to_numeric(l[6]) +pd.to_numeric(l[7]) +pd.to_numeric(l[8]) +pd.to_numeric(l[9]) +pd.to_numeric(l[10]) +pd.to_numeric(l[11]) +pd.to_numeric(l[12]) +pd.to_numeric(l[13]) +pd.to_numeric(l[14]) +pd.to_numeric(l[15]) +pd.to_numeric(l[16]) +pd.to_numeric(l[17]) )/len(l)
    
l = list([data.TIME008, data.TIME009, data.TIME010, data.TIME011, data.TIME012, data.TIME013, data.TIME016, data.TIME017, data.TIME018, data.TIME019, data.TIME020, data.TIME021, data.TIME024, data.TIME025, data.TIME026, data.TIME027, data.TIME028, data.TIME029])
data['av_time_review']=data.apply(lambda x:av_time_review(l))

# Get the responses given to VQ's
def response_vq(response,id_question):
    if math.isnan(response): return 'none'
    else: return ((validation_q[(validation_q.ID_VQ==id_question) & (validation_q.VALUE_RESPONSE==response)]['TEXT_RESPONSE']).values)[0]

# this line is not working
data['rex_vq_1'] = data.VQ04.apply(lambda x:response_vq(pd.to_numeric(x),'VQ04'))
data['rex_vq_2'] = data.VQ05.apply(lambda x:response_vq(pd.to_numeric(x),'VQ05'))
data['castle_vq_1'] = data.VQ06.apply(lambda x:response_vq(pd.to_numeric(x),'VQ06'))
data['castle_vq_2'] = data.VQ07.apply(lambda x:response_vq(pd.to_numeric(x),'VQ07'))
data['omni_vq_1'] = data.VQ08.apply(lambda x:response_vq(pd.to_numeric(x),'VQ08'))
data['omni_vq_2'] = data.VQ09.apply(lambda x:response_vq(pd.to_numeric(x),'VQ09'))

def facts_rate(id_review):
    facts=codes[codes.Review_id_ArguAna==id_review].n_facts.values[0]
    opinions=codes[codes.Review_id_ArguAna==id_review].n_negative.values[0]+codes[codes.Review_id_ArguAna==id_review].n_positive.values[0]
    return facts/opinions
def negative_op_rate(id_review):
    negative_op=codes[codes.Review_id_ArguAna==id_review].n_negative.values[0]
    opinions=codes[codes.Review_id_ArguAna==id_review].n_negative.values[0]+codes[codes.Review_id_ArguAna==id_review].n_positive.values[0]
    return negative_op/opinions
def positive_op_rate(id_review):
    positive_op=codes[codes.Review_id_ArguAna==id_review].n_positive.values[0]
    opinions=codes[codes.Review_id_ArguAna==id_review].n_negative.values[0]+codes[codes.Review_id_ArguAna==id_review].n_positive.values[0]
    return positive_op/opinions
#facts_rate(869)
codes['facts_rate']=codes.Review_id_ArguAna.apply(lambda x: facts_rate(x))
codes['negative_op_rate']=codes.Review_id_ArguAna.apply(lambda x: negative_op_rate(x))
codes['positive_op_rate']=codes.Review_id_ArguAna.apply(lambda x: positive_op_rate(x))
#TODO: Outliers of times per review
c=codes.Review_id_ArguAna

# =============================================================================
# Apply the rules to reject----------------------------------------------------
# =============================================================================
data["TIME_SUM"] = pd.to_numeric(data["TIME_SUM"])

data_reject=data[(data['TIME_SUM']<6*60) | (data['av_time_review']<15) | (data['correct_responses']<=2)]
data_approved=data[(data['TIME_SUM']>=6*60) & (data['av_time_review']>=15) & (data['correct_responses']>2)]


data_reject=data_reject[['CASE','WI01_01','correct_responses','av_time_review','TIME_SUM','R204_09','R204_10','rex_vq_1','rex_vq_2','castle_vq_1','castle_vq_2','omni_vq_1','omni_vq_2','VQ04','VQ05','VQ06','VQ07','VQ08','VQ09','TIME008','TIME009','TIME010','TIME011','TIME012','TIME013','TIME016','TIME017','TIME018','TIME019','TIME020','TIME021','TIME024','TIME025','TIME026','TIME027','TIME028','TIME029']]
data_approved=data_approved[['CASE','WI01_01','correct_responses','av_time_review','TIME_SUM','R204_09','R204_10','rex_vq_1','rex_vq_2','castle_vq_1','castle_vq_2','omni_vq_1','omni_vq_2','VQ04','VQ05','VQ06','VQ07','VQ08','VQ09','TIME008','TIME009','TIME010','TIME011','TIME012','TIME013','TIME016','TIME017','TIME018','TIME019','TIME020','TIME021','TIME024','TIME025','TIME026','TIME027','TIME028','TIME029']]

len(data_approved['CASE'])
len(data_reject['CASE'])

# =============================================================================
# Prepare dataset to work with------------------------------------------------------------------------
# =============================================================================
# Temporary, automatic selection--------------------
#df=pd.merge(data, data_approved, how='inner') #TODO: hacer esto de manera más elegante!!

# Final: Using final cases documented in Google drive
finalCases = pd.read_csv('CasesFinal.csv')
casesIds=np.array(finalCases.CASE)
casesIds=casesIds.astype(str)
df=data[data.CASE.isin(casesIds)]

# rename columns
cols = dict()
cols_int= dict()
reasons_suffix = np.array(['01','02','03','04','05','06','07','08','09','10'])
for i, row in codes.iterrows():
    #ID_SoSci_Helpfulness -> ID_A_Helpfulness
    cols[row['ID_SoSci_Helpfulness']+'_01'] = row['ID_A_Helpfulness']
    cols_int[row['ID_A_Helpfulness']] = 'int' 
    for j in reasons_suffix:
        #ID_SoSci_Reasons -> ID_A_Reasons
        cols[row['ID_SoSci_Reasons']+'_'+j] = row['ID_A_Reasons'][0:6]+j
        cols_int[row['ID_A_Reasons'][0:6]+j] = 'int'

df.rename(columns=cols, inplace=True)
#data = data.query('age >18').query('age < 32')

#Change type variables
df = df.astype(cols_int)
# Add ArguAna values
#def get_AA_value(ID_A_Helpfulness, rate_type):
#    return codes[codes.ID_A_Helpfulness==ID_A_Helpfulness][rate_type].values[0]
#for i in codes['ID_A_Helpfulness']:
#    df[i[0:5]+'FR'] = get_AA_value(i,'facts_rate')


#get_AA_value('OR03_H','negative_op_rate')
#get_AA_value('OR03_H','positive_op_rate')

# Get means helpfulness
h_means = np.array([[0,0]])
h_rates = np.array([[0,0]])
h_medians = np.array([[0,0]])
h_modes = np.array([[0,0]])
for i,row in codes.iterrows():
    h_means=np.concatenate((h_means, np.array([[row['ID_A_Helpfulness'], round(df[row['ID_A_Helpfulness']].mean(),2)]])))
    h_medians=np.concatenate((h_medians, np.array([[row['ID_A_Helpfulness'], round(df[row['ID_A_Helpfulness']].median(),2)]])))
    h_modes=np.concatenate((h_modes, np.array([[row['ID_A_Helpfulness'], round(df[row['ID_A_Helpfulness']].mode()[0],2)]])))
    h_counts=df[row['ID_A_Helpfulness']].value_counts()
    h_rate=h_counts[4]+h_counts[5]/sum(h_counts)
    h_rates=np.concatenate((h_rates, np.array([[row['ID_A_Helpfulness'], round(h_rate,2)]])))
# df[row['ID_A_Helpfulness']].mode()[0]   
#
#
#d=[1,1,1,1,1,1,2,3,1,1,1,8]
#d1=pd.DataFrame(d)
#d1.mode()[0][0].values()
#np.mean(d)
#np.median(d)
#np.mean(d)

h_means=h_means[1:,:]
h_rates=h_rates[1:,:]
#h_m_r=np.hstack(h_means,h_rates)
h_means=pd.DataFrame(h_means).sort_values([1])
h_rates=pd.DataFrame(h_rates).sort_values([1])

#h_medians=h_medians[1:,:]
#h_medians=pd.DataFrame(h_medians).sort_values([1])
#

#fig = plt.figure()
#plt.title('Mean response Helpfulness')
#plt.plot(h_means[0],h_means[1])

#data_approved[data_approved.CASE=='610']
#data_reject[data_reject.CASE=='610']

# =============================================================================
#--------------Merge all reviews --------------------------------------------
#--------------LogReg and SVM--------------------------------------------
# =============================================================================
from sklearn import metrics
# Case 1: Spearman's Correlation
vbles=['H','E01','E02','E03','E04','E05','E06','E07','E08','E09','E10']
df_all = pd.DataFrame(columns=vbles)
df_aux = pd.DataFrame(columns=vbles)
for i in codes['ID_A_Helpfulness']:
    for j in vbles:
        df_aux[j]=df[i[0:5]+j]
    #df_aux['ID_REV']=i
    df_all=df_all.append(df_aux, ignore_index = True)
df_all=df_all.astype(float)
df_all_corr=df_all.corr(method='spearman')

# case 2: LogReg binary
ybin = df_all['H'].copy()
ybin[(ybin==1)|(ybin==2)|(ybin==3)]=0 # non H at all, non H, neutral
ybin[(ybin==4)|(ybin==5)]=1 #H, vey H
X=df_all[vbles[1:]]

# Split
X_train,X_test,y_train,y_test=train_test_split(X,ybin,test_size=0.25,random_state=0)
# LogitReg 
logreg = LogisticRegression(multi_class='multinomial',solver ='newton-cg')
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

coef_models2=pd.DataFrame(logreg.coef_)
coef_models2.columns=['E01','E02','E03','E04','E05','E06','E07','E08','E09','E10']
acc=metrics.accuracy_score(y_test, y_pred)
print('Acc LogitReg Bin, all reviews: '+str(acc))
f1=metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print('F1 LogitReg Bin, all reviews: '+str(f1))

# class weighting for unbalanced classes 
logreg = LogisticRegression(multi_class='multinomial',solver ='newton-cg', random_state=0, class_weight='balanced')
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)


# prueba de pseudo r2
aux=np.ones(y_test.size)
intercept=aux*logreg.intercept_
logreg_intercept = LogisticRegression(multi_class='multinomial',solver ='newton-cg', random_state=0, class_weight='balanced')
logreg_intercept.fit(intercept.reshape(-1, 1),y_test)

y_prob = logreg.predict_proba(X_test)
y_prob_intercept = logreg_intercept.predict_proba(intercept.reshape(-1, 1))
# keep the predictions for class 1 only
y_prob = y_prob[:, 1]
y_prob_intercept = y_prob_intercept[:, 1] 

r2= 1-(sum(np.log(y_prob))/sum(np.log(y_prob_intercept)))
loss = log_loss(y_test, y_prob)
np.sum(r2)

# using log loss
y_prob = logreg.predict_proba(X_test)
loss_model_coeff=log_loss(y_test, y_prob)    # 0.5317

y_prob_intercept = logreg_intercept.predict_proba(intercept.reshape(-1, 1))
loss_model_intercept=log_loss(y_test, y_prob_intercept)    # 0.69

r2 = 1-(loss_model_coeff/loss_model_intercept) #0.23 -> value reported in paper!!

#---


coef_models2balanced=pd.DataFrame(logreg.coef_)
coef_models2balanced.columns=['E01','E02','E03','E04','E05','E06','E07','E08','E09','E10']
acc=metrics.accuracy_score(y_test, y_pred)
print('Acc LogitReg Bin, all reviews, balanced classes: '+str(acc))
f1=metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print('F1 LogitReg Bin, all reviews, balanced classes: '+str(f1))
r2=metrics.r2_score(y_test, y_pred)
print('R2 LinReg Bin all reviews: '+str(r2))

#------- signifficance 

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

def logit_pvalue(model, x):
    """ Calculate z-scores for scikit-learn LogisticRegression.
    parameters:
        model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
        x:     matrix on which the model was fit
    This function uses asymtptics for maximum likelihood estimates.
    """
    p = model.predict_proba(x)
    n = len(p)
    m = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    ans = np.zeros((m, m))
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se  
    p = (1 - norm.cdf(abs(t))) * 2
    return p

# test p-values
logit_pvalues=logit_pvalue(logreg, X_train)

reasons_names=['Lenght','Details','Objectivity','Pros&Cons','Convincing','Aspects','FlowArg','Credibility','Emotional','Episodic']
alpha1=0.05
alpha2=0.01
alpha3=0.001

logit_pvalues_df=pd.DataFrame({'reasons':reasons_names, 'p_values':logit_pvalues[1:].tolist(), 'significant_05':logit_pvalues[1:]<alpha1, 'significant_01':logit_pvalues[1:]<alpha2, 'significant_001':logit_pvalues[1:]<alpha3})
logit_pvalues_df['coef']=logreg.coef_[0]
logit_pvalues_df.set_index('reasons')
#logit_pvalues_df[['reasons','p_values']].plot()
#logit_pvalues_df.transpose().plot()

# compare with statsmodels
import statsmodels.api as sm
sm_model = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=0)
print(sm_model.pvalues)
sm_model.summary()


OR=np.exp(sm_model.params)





#------end significance



# case 3: SVM
clf = svm.SVC()
clf.fit(X_train,y_train)
scores = cross_val_score(clf, X, ybin, cv=5)
accSVM=scores.mean()
print('Acc SVM, all reviews: '+str(accSVM))
scores = cross_val_score(clf, X, ybin, cv=5,scoring='f1_macro')
F1SVM=scores.mean()
print('F1 SVM, all reviews: '+str(F1SVM))

# class weighting for unbalanced classes 
clf = svm.SVC(class_weight='balanced', C=1.0, random_state=0)
clf.fit(X_train,y_train)
scores = cross_val_score(clf, X, ybin, cv=5)
accSVM=scores.mean()
print('Acc SVM, all reviews, class weight balance: '+str(accSVM))
scores = cross_val_score(clf, X, ybin, cv=5,scoring='f1_macro')
F1SVM=scores.mean()
print('F1 SVM, all reviews, class weight balance: : '+str(F1SVM))

# case 4: LogReg multiclass
y = df_all['H'].copy()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
logreg = LogisticRegression(multi_class='multinomial',solver ='newton-cg')
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
accMc=metrics.accuracy_score(y_test, y_pred)
print('Acc LogitReg MC unbalanced: '+str(accMc))
f1=metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average=None, sample_weight=None)
print('F1 LogitReg MC, unbalanced: '+str(f1))

logreg = LogisticRegression(multi_class='multinomial',solver ='newton-cg',random_state=0, class_weight='balanced')
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
accMc=metrics.accuracy_score(y_test, y_pred)
print('Acc LogitReg MC balanced: '+str(accMc))
f1=metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average=None, sample_weight=None)
print('F1 LogitReg MC, balanced: '+str(f1))

y.value_counts()
ybin.value_counts()

#correlation matrix, showing correlation between each variable and all the others -------------------
import seaborn as sns
reasons_names=['Lenght','Details','Objectivity','Pros&Cons','Convincing','Aspects','FlowArg','Credibility','Emotional','Episodic']
X.columns=reasons_names
Xcorr = X.corr() 
X.corr().head()
sns.heatmap(Xcorr, cmap = 'bwr')

# test corr between details and Lenght is significantly diff from the rest of vbles
hcorr_value = np.ones(8)*Xcorr['Lenght']['Details']
col_corr=Xcorr['Lenght'][2:]
stat, p = ttest_ind(hcorr_value, col_corr)
alpha1=0.05
alpha2=0.01
alpha3=0.001
p<alpha3

hcorr_value = np.ones(7)*Xcorr['Lenght']['Objectivity']
others_corr=Xcorr['Lenght'][3:]

XcorrAux=Xcorr 
hcorr_value = np.ones(8)*Xcorr['Lenght']['Objectivity']
others_corr=Xcorr.drop(['Lenght', 'Objectivity'])['Lenght']
stat, p = ttest_ind(hcorr_value, others_corr)
alpha1=0.05
alpha2=0.01
alpha3=0.001
p<alpha3


#--------------Mediator variables analysis: credibility--------------------------------------------
# vbles=['H','E01','E02','E03','E04','E05','E06','E07','E08','E09','E10']

#Step 1: Regress the dependent variable on the independent variable to confirm that the independent variable is a significant predictor of the dependent variable.
ybin = df_all['H'].copy()
ybin[(ybin==1)|(ybin==2)|(ybin==3)]=0 # strongly disagree, disagree, neutral
ybin[(ybin==4)|(ybin==5)]=1 # agree, totally agree
vbles=['E01','E02','E03','E04','E06','E07','E09','E10']# all but convincing and credible
#vbles=['E01','E02','E03','E04','E05','E06','E07','E08','E09','E10'] 
X=df_all[vbles]

# Split
X_train,X_test,y_train,y_test=train_test_split(X,ybin,test_size=0.25,random_state=0)
logreg = LogisticRegression(multi_class='multinomial',solver ='newton-cg', random_state=0, class_weight='balanced')
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
reasons_names=['Lenght','Details','Objectivity','Pros&Cons','Aspects','FlowArg','Emotional','Episodic']
#reasons_names=['Lenght','Details','Objectivity','Pros&Cons','Convincing','Aspects','FlowArg','Credibility','Emotional','Episodic']

# test p-values
logit_pvalues=logit_pvalue(logreg, X_train)
alpha1=0.05
alpha2=0.01
alpha3=0.001

logit_pvalues_df=pd.DataFrame({'reasons':reasons_names, 'p_values':logit_pvalues[1:].tolist(), 'significant_05':logit_pvalues[1:]<alpha1, 'significant_01':logit_pvalues[1:]<alpha2, 'significant_001':logit_pvalues[1:]<alpha3})
logit_pvalues_df['coef']=logreg.coef_[0]

# Step 2: Regress the mediator on the independent variable to confirm that the independent variable is a significant predictor of the mediator. If the mediator is not associated with the independent variable, then it couldn’t possibly mediate anything.
ybin = df_all['E08'].copy()
ybin[(ybin==1)|(ybin==2)|(ybin==3)]=0 # strongly disagree, disagree, neutral
ybin[(ybin==4)|(ybin==5)]=1 # agree, totally agree
vbles=['E01','E02','E03','E04','E06','E07','E09','E10']
X=df_all[vbles]

# Split
X_train,X_test,y_train,y_test=train_test_split(X,ybin,test_size=0.25,random_state=0)
# class weighting for unbalanced classes 
logreg = LogisticRegression(multi_class='multinomial',solver ='newton-cg', random_state=0, class_weight='balanced')
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

reasons_names=['Lenght','Details','Objectivity','Pros&Cons','Aspects','FlowArg','Emotional','Episodic']
#coef_models2balanced=pd.DataFrame(logreg.coef_)
#coef_models2balanced.columns=reasons_names
#acc=metrics.accuracy_score(y_test, y_pred)
#print('Acc LogitReg Bin, all reviews, balanced classes: '+str(acc))
#f1=metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
#print('F1 LogitReg Bin, all reviews, balanced classes: '+str(f1))
#
#coef_models2balanced.plot.bar(title='Regression coefficients - Credibility', xlim=[-0.2, 0.5])

# test p-values
logit_pvalues=logit_pvalue(logreg, X_train)
alpha1=0.05
alpha2=0.01
alpha3=0.001

logit_pvalues_df=pd.DataFrame({'reasons':reasons_names, 'p_values':logit_pvalues[1:].tolist(), 'significant_05':logit_pvalues[1:]<alpha1, 'significant_01':logit_pvalues[1:]<alpha2, 'significant_001':logit_pvalues[1:]<alpha3})
logit_pvalues_df['coef']=logreg.coef_[0]

# Step 3: Regress the dependent variable on both the mediator and independent variable to confirm that the mediator is a significant predictor of the dependent variable, and the previously significant independent variable in Step #1 is now greatly reduced, if not nonsignificant.

ybin = df_all['H'].copy()
ybin[(ybin==1)|(ybin==2)|(ybin==3)]=0 # strongly disagree, disagree, neutral
ybin[(ybin==4)|(ybin==5)]=1 # agree, totally agree
vbles=['E01','E02','E03','E04','E06','E07','E09','E10','E08']# all but convincing 
#vbles=['E01','E02','E03','E04','E05','E06','E07','E08','E09','E10'] 
X=df_all[vbles]

X_train,X_test,y_train,y_test=train_test_split(X,ybin,test_size=0.25,random_state=0)
logreg = LogisticRegression(multi_class='multinomial',solver ='newton-cg', random_state=0, class_weight='balanced')
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
reasons_names=['Lenght','Details','Objectivity','Pros&Cons','Aspects','FlowArg','Emotional','Episodic','Credibility']
#reasons_names=['Lenght','Details','Objectivity','Pros&Cons','Convincing','Aspects','FlowArg','Credibility','Emotional','Episodic']

# test p-values
logit_pvalues=logit_pvalue(logreg, X_train)
alpha1=0.05
alpha2=0.01
alpha3=0.001

logit_pvalues_df=pd.DataFrame({'reasons':reasons_names, 'p_values':logit_pvalues[1:].tolist(), 'significant_05':logit_pvalues[1:]<alpha1, 'significant_01':logit_pvalues[1:]<alpha2, 'significant_001':logit_pvalues[1:]<alpha3})
logit_pvalues_df['coef']=logreg.coef_[0]



#--------------Mediator variables analysis: convincing--------------------------------------------
# Step 2: Regress the mediator on the independent variable to confirm that the independent variable is a significant predictor of the mediator. If the mediator is not associated with the independent variable, then it couldn’t possibly mediate anything.
ybin = df_all['E05'].copy()
ybin[(ybin==1)|(ybin==2)|(ybin==3)]=0 # strongly disagree, disagree, neutral
ybin[(ybin==4)|(ybin==5)]=1 # agree, totally agree
vbles=['E01','E02','E03','E04','E06','E07','E09','E10']
X=df_all[vbles]

# Split
X_train,X_test,y_train,y_test=train_test_split(X,ybin,test_size=0.25,random_state=0)
# class weighting for unbalanced classes 
logreg = LogisticRegression(multi_class='multinomial',solver ='newton-cg', random_state=0, class_weight='balanced')
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

reasons_names=['Lenght','Details','Objectivity','Pros&Cons','Aspects','FlowArg','Emotional','Episodic']
#coef_models2balanced=pd.DataFrame(logreg.coef_)
#coef_models2balanced.columns=reasons_names
#acc=metrics.accuracy_score(y_test, y_pred)
#print('Acc LogitReg Bin, all reviews, balanced classes: '+str(acc))
#f1=metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
#print('F1 LogitReg Bin, all reviews, balanced classes: '+str(f1))
#
#coef_models2balanced.plot.bar(title='Regression coefficients - Credibility', xlim=[-0.2, 0.5])

# test p-values
logit_pvalues=logit_pvalue(logreg, X_train)
alpha1=0.05
alpha2=0.01
alpha3=0.001

logit_pvalues_df=pd.DataFrame({'reasons':reasons_names, 'p_values':logit_pvalues[1:].tolist(), 'significant_05':logit_pvalues[1:]<alpha1, 'significant_01':logit_pvalues[1:]<alpha2, 'significant_001':logit_pvalues[1:]<alpha3})
logit_pvalues_df['coef']=logreg.coef_[0]

# Step 3: Regress the dependent variable on both the mediator and independent variable to confirm that the mediator is a significant predictor of the dependent variable, and the previously significant independent variable in Step #1 is now greatly reduced, if not nonsignificant.

ybin = df_all['H'].copy()
ybin[(ybin==1)|(ybin==2)|(ybin==3)]=0 # strongly disagree, disagree, neutral
ybin[(ybin==4)|(ybin==5)]=1 # agree, totally agree
vbles=['E01','E02','E03','E04','E06','E07','E09','E10','E05']# all but convincing 
#vbles=['E01','E02','E03','E04','E05','E06','E07','E08','E09','E10'] 
X=df_all[vbles]

X_train,X_test,y_train,y_test=train_test_split(X,ybin,test_size=0.25,random_state=0)
logreg = LogisticRegression(multi_class='multinomial',solver ='newton-cg', random_state=0, class_weight='balanced')
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
reasons_names=['Lenght','Details','Objectivity','Pros&Cons','Aspects','FlowArg','Emotional','Episodic','Convincing']
#reasons_names=['Lenght','Details','Objectivity','Pros&Cons','Convincing','Aspects','FlowArg','Credibility','Emotional','Episodic']

# test p-values
logit_pvalues=logit_pvalue(logreg, X_train)
alpha1=0.05
alpha2=0.01
alpha3=0.001

logit_pvalues_df=pd.DataFrame({'reasons':reasons_names, 'p_values':logit_pvalues[1:].tolist(), 'significant_05':logit_pvalues[1:]<alpha1, 'significant_01':logit_pvalues[1:]<alpha2, 'significant_001':logit_pvalues[1:]<alpha3})
logit_pvalues_df['coef']=logreg.coef_[0]

# =============================================================================
#-----------------VALIDATION OF DATA
# =============================================================================

reasons_suffix = np.array(['01','02','03','04','05','06','07','08','09','10'])
reasons_values = np.array([])
data_values = pd.DataFrame(columns=['worker_id','case','id_review', 'RevText','Helpfulness','Reason1','Reason2','Reason3','Reason4','Reason5','Reason6','Reason7','Reason8','Reason9','Reason10'])
survey_values = pd.DataFrame(columns=['worker_id','case','started', 'end','Age','Gender','EducationLevel','InternetHabitsShopping','InternetHabitsHotel','InternetHabitsRestaurant','InternetHabitsMovie','ValidationCastle', 'ValidationRex', 'ValidationOmni',
'Aspect1','Aspect2','Aspect3','Aspect4','Aspect5','Aspect6','Aspect7','Aspect8','Aspect9','Aspect10','Aspect11'])
time_values = pd.DataFrame(columns=['worker_id','time001','time002','time003','time004','time005','time007','time008','time009','time010','time011','time012','time013','time014','time015','time016','time017','time018','time019','time020','time021','time022','time023','time024','time025','time026','time027','time028','time029','time030','time031'])



# Loop data
for index_d, row_d in data.iterrows():
    # loop reviews
    case = row_d['CASE']
    survey_values = survey_values.append(pd.DataFrame({'worker_id':[row_d['WI01_01']],'case':[row_d['CASE']],'started':[row_d['STARTED']],'end':[row_d['LASTDATA']],'Age':[row_d['DI01_01']],'Gender':[row_d['DI02']],'EducationLevel':[row_d['DI03']],'InternetHabitsShopping':[row_d['DI08']],'InternetHabitsHotel':[row_d['DI05']],'InternetHabitsRestaurant':[row_d['DI06']],
                                                       'InternetHabitsMovie':[row_d['DI07']],'ValidationCastle':[row_d['VQ01']],'ValidationRex':[row_d['VQ02']],'ValidationOmni':[row_d['VQ03']],
                                                       'Aspect1':[row_d['AI01_01']],'Aspect2':[row_d['AI01_02']],'Aspect3':[row_d['AI01_03']],'Aspect4':[row_d['AI01_04']],'Aspect5':[row_d['AI01_05']],'Aspect6':[row_d['AI01_06']],'Aspect7':[row_d['AI01_07']],'Aspect8':[row_d['AI01_08']],'Aspect9':[row_d['AI01_09']],'Aspect10':[row_d['AI01_10']],'Aspect11':[row_d['AI01_11']] }))
                               
    time_values = time_values.append(pd.DataFrame({'worker_id':[row_d['WI01_01']],'case':[row_d['CASE']],
                                                    'time001':[row_d['TIME001']],'time002':[row_d['TIME002']],'time003':[row_d['TIME003']],'time004':[row_d['TIME004']],'time005':[row_d['TIME005']],'time007':[row_d['TIME007']],'time008':[row_d['TIME008']],'time009':[row_d['TIME009']],'time010':[row_d['TIME010']],'time011':[row_d['TIME011']],'time012':[row_d['TIME012']],'time013':[row_d['TIME013']],'time014':[row_d['TIME014']],'time015':[row_d['TIME015']],'time016':[row_d['TIME016']],'time017':[row_d['TIME017']]
                                                    ,'time018':[row_d['TIME018']],'time019':[row_d['TIME019']],'time020':[row_d['TIME020']],'time021':[row_d['TIME021']],'time022':[row_d['TIME022']],'time023':[row_d['TIME023']],'time024':[row_d['TIME024']],'time025':[row_d['TIME025']],'time026':[row_d['TIME026']],'time027':[row_d['TIME027']],'time028':[row_d['TIME028']],'time029':[row_d['TIME029']],'time030':[row_d['TIME030']],'time031':[row_d['TIME031']]}))
    for index, row in codes.iterrows():
        ##print(row['Review_text'][0:20])
        rev_text = row['Review_text'][0:30]
        ID_SoSci_Helpfulness = row['ID_SoSci_Helpfulness']+'_01'
        ID_SoSci_Reasons = row['ID_SoSci_Reasons']
        id_review = row['ID_A_Helpfulness'][0:4]
        #print(ID_SoSci_Helpfulness)
        #print(ID_SoSci_Reasons)
        helpfulness_value = row_d[ID_SoSci_Helpfulness]
        reasons_values = []
        for j in reasons_suffix:
             #print(row['ID_SoSci_Reasons']+'_'+j)
             reasons_values.append(row_d[ID_SoSci_Reasons+'_'+j])    
        data_values = data_values.append(pd.DataFrame({'worker_id':[row_d['WI01_01']],'case':[case],'id_review':[id_review], 'RevText':[rev_text],'Helpfulness':[helpfulness_value],'Reason1':[reasons_values[0]]
        ,'Reason2':[reasons_values[1]],'Reason3':[reasons_values[2]],'Reason4':[reasons_values[3]],'Reason5':[reasons_values[4]]
        ,'Reason6':[reasons_values[5]],'Reason7':[reasons_values[6]],'Reason8':[reasons_values[7]],'Reason9':[reasons_values[8]],'Reason10':[reasons_values[9]]}))
# data_values=data_values.sort_values(data_values['case'])
    
##------------------------------------
# Loop data
for index_d, row_d in data.iterrows():
    # loop reviews
    case = row['ID_SoSci_Reasons']
    for index, row in codes.iterrows():
        ##print(row['ID_SoSci_Reasons']) 
        rev_text = row['Review_text'][0:30]
        print(rev_text)
        ID_SoSci_Helpfulness = row['ID_SoSci_Helpfulness']+'_01'
        ID_SoSci_Reasons = row['ID_SoSci_Reasons']
        ##ID_SoSci_Helpfulness = 'R010_02'
        helpfulness_value = row_d[ID_SoSci_Helpfulness]
        #print(helpfulness_value)
        # Loop reasons
        reasons_values = []
        for j in reasons_suffix:
             #print(row['ID_SoSci_Reasons']+'_'+j)
             reasons_values.append(row_d[ID_SoSci_Reasons+'_'+j])
             ##print(row['ID_A_Reasons'][0:6]+j)
    data_values = data_values.append(pd.DataFrame({'case':[case],'RevText':[rev_text],'Helpfulness':[helpfulness_value],'Reason1':[reasons_values[0]]
    ,'Reason2':[reasons_values[1]],'Reason3':[reasons_values[2]],'Reason4':[reasons_values[3]],'Reason5':[reasons_values[4]]
    ,'Reason6':[reasons_values[5]],'Reason7':[reasons_values[6]],'Reason8':[reasons_values[7]],'Reason9':[reasons_values[8]],'Reason10':[reasons_values[9]]}))
                                                   
# =============================================================================
# Statistics and plots -----------------------------------------------
# =============================================================================


# Plot histograms of helpfulness and reasons replies
for i,row in codes.iterrows():
    fig = plt.figure()
    plt.title(row['ID_A_Helpfulness'])
    #plt.hist(df[row['ID_A_Helpfulness']])
    df[row['ID_A_Helpfulness']].value_counts().sort_index().plot.bar()
#    for j in reasons_suffix:
#        fig = plt.figure()
#        plt.title(row['ID_A_Reasons'][0:6]+j)
#        plt.hist(df[row['ID_A_Reasons'][0:6]+j])
#def get_text(code):
#    return (codes[codes['ID_A_Helpfulness']==code]['Review_text'].values)[0]
#
#
## this line is not working
#h_means['txt_len'] = h_means['txt'].apply(lambda x:len(x))


# check normality
from scipy import stats

for i in codes['ID_A_Helpfulness']:
    k2, p = stats.normaltest(df[i])
    alpha = 1e-3
    #print("p = {:g}".format(p))
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("The null hypothesis can be rejected: "+i+", p = {:g}".format(p))
    else:
        print("The null hypothesis cannot be rejected: "+i+", p = {:g}".format(p))



# =============================================================================
#--------------Regression ----------------------------------------------
# =============================================================================

#split dataset in features and target variable
feature_cols = ['OR03_E01', 'OR03_E02', 'OR03_E03', 'OR03_E04', 'OR03_E05', 'OR03_E06', 'OR03_E07', 'OR03_E08', 'OR03_E09', 'OR03_E10']
#feature_cols = ['OR03_E01', 'OR03_E02', 'OR03_E03',  'OR03_E05', 'OR03_E06', 'OR03_E07', 'OR03_E08'] #mejor combinacion
#feature_cols = ['CR01_E01', 'CR01_E02', 'CR01_E03', 'CR01_E04', 'CR01_E05', 'CR01_E06', 'CR01_E07', 'CR01_E08', 'CR01_E09', 'CR01_E10']
X = df[feature_cols] # Features
y = df.OR03_H.copy() # Target variable
#y = df.CR01_H.copy() # Target variable
y.value_counts()
#y = df.CR01_H # Target variable

# check linearity per variable  ------------------------------------
X_aux = df[feature_cols_aux] # Features
X_aux = X_aux.sort_values('OR03_H')
feature_cols_aux=feature_cols.copy()
feature_cols_aux.append('OR03_H')
fig = plt.figure()
plt.title('Linearity')
plt.scatter(X_aux.OR03_E01, X_aux.OR03_H)


from sklearn.cross_validation import train_test_split

# manipulate y vble
#y = df.OR03_H.copy() # Target variable  # si no haces la copia, se sobreescribirían la variable df originalc con la siguiente linea :/
#y[(y==1)|(y==2)|(y==3)|(y==4)]=0
#y[(y==5)]=1
#y.value_counts()
# end manipulation y vble


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# import the class
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model

# instantiate the model (using the default parameters)
logreg = LogisticRegression(multi_class='multinomial',solver ='newton-cg')
#logreg = LogisticRegression()
#logreg = LogisticRegression(multi_class='ovr',solver ='liblinear')
#logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)

# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

import seaborn as sns

# Print confusion matrix 1st version
class_names=[2,3,4,5] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# Print confusion matrix 2nd version
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True , cmap="YlGnBu" ,fmt='g')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average=None))
print("Recall:",metrics.recall_score(y_test, y_pred,average=None))
coef=logreg.coef_
logreg.classes_
coef[0,:].sum()

# PCA + Logistic regression -----------------------------------------------------------------
import sklearn.decomposition as skdc ##Includes Principal Component Analysis, a method of dimensionality reduction
import sklearn.pipeline as skpl ##Convenient module for calculating PCs and using them in logistic regression

#correlation matrix, showing correlation between each variable and all the others ------
Xcorr = X.corr() 
X.corr().head()
sns.heatmap(Xcorr, cmap = 'bwr')

pca = skdc.PCA()
pcafit = pca.fit_transform(X,y) ##apply dimensionality reduction to X

var_explained = pca.explained_variance_ratio_ #ratio of variance each PC explains
print(pd.Series(var_explained))
var_explained[0:8].sum()
# 8 componentes explican al menos el 95% de la varianza

pca = skdc.PCA(n_components = 9) #only include first 10 components
logreg = LogisticRegression(multi_class='multinomial',solver ='newton-cg')#empty model space
pipeline = skpl.Pipeline([('pca', pca), ('logistic', logreg)]) #create pipeline from pca to logregression space

# corregir esto!! esta doble asignación hace que lo que se almacene en una vble se quede en la otra
pred1Right=pred1Wrong=0
pred2Right=pred2Wrong=0
pred3Right=pred3Wrong=0
pred4Right=pred4Wrong=0
pred5Right=pred5Wrong=0

for i in range(0,110): #run through each row in data set
    trainX = X.drop(i, 0) #train model with predictor dataframe, remove single row
    trainy = y.drop(i,0) #train model with response array, remove single row
    testX = X.iloc[i,:].values.reshape(1,10) #Removed row will be test predictor (Got error message before using values.reshape)
    testy = y[i] #Removed value will be test response
    fit = pipeline.fit(trainX, trainy) #fit model
    prediction = pipeline.predict(testX) #test model with left out value
    #print(str(prediction)+'-'+str(testy))
    if prediction == 1 and testy == 1:
        pred1Right += 1
    elif prediction == 1 and testy != 1:
        pred1Wrong += 1
    elif prediction == 2 and testy == 2:
        pred2Right += 1
    elif prediction == 2 and testy != 2:
        pred2Wrong += 1
    elif prediction == 3 and testy == 3:
        pred3Right += 1
    elif prediction == 3 and testy != 3:
        pred3Wrong += 1
    elif prediction == 4 and testy == 4:
        pred4Right += 1
    elif prediction == 4 and testy != 4:
        pred4Wrong += 1
    elif prediction == 5 and testy == 5:
        pred5Right += 1
    elif prediction == 5 and testy != 5:
        pred5Wrong += 1

mis = (pred1Wrong+pred2Wrong+pred3Wrong+pred4Wrong+pred5Wrong)/(pred1Wrong+pred2Wrong+pred3Wrong+pred4Wrong+pred5Wrong+pred1Right+pred2Right+pred3Right+pred5Right+pred5Right) #calculate misclassification rate
# 45% de misclasification con PCA+LogReg (multinomial y no multinomial es lo mismo), 39% con solo LogReg

# Linear regression----------------------------------------------
feature_cols = ['OR03_E01', 'OR03_E02', 'OR03_E03',  'OR03_E05', 'OR03_E06', 'OR03_E07', 'OR03_E08'] #mejor combinacion
#feature_cols = ['CR01_E01', 'CR01_E02', 'CR01_E03', 'CR01_E04', 'CR01_E05', 'CR01_E06', 'CR01_E07', 'CR01_E08', 'CR01_E09', 'CR01_E10']
X = df[feature_cols] # Features
y = df.OR03_H.copy() # Target variable
#y = df.CR01_H.copy() # Target variable
#y = df.CR01_H # Target variable
from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# import the class
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model

# instantiate the model (using the default parameters)
linreg = linear_model.LinearRegression()

# fit the model with data
linreg.fit(X_train,y_train)

#
y_pred=linreg.predict(X_test)
linreg.score(X,y)

import math
#y_hat = model.predict(X)
#AIC=2k-2ln(L), L: the maximum value of the likelihood function
resid = y_test - y_pred
sse = sum(resid**2)
k= 10
AIC= 2*k - 2*math.log(sse)


# Firth regression-------------------------------------------------------------
def firth_likelihood(beta, logit):
    return -(logit.loglike(beta) + 0.5*np.log(np.linalg.det(-logit.hessian(beta))))

# Do firth regression
# Note information = -hessian, for some reason available but not implemented in statsmodels
def fit_firth(y, X, start_vec, step_limit=1000, convergence_limit=0.0001):

    logit_model = smf.Logit(y, X)
    
    if start_vec is None:
        start_vec = np.zeros(X.shape[1])
    
    beta_iterations = []
    beta_iterations.append(start_vec)
    for i in range(0, step_limit):
        pi = logit_model.predict(beta_iterations[i])
        W = np.diagflat(np.multiply(pi, 1-pi))
        var_covar_mat = np.linalg.pinv(-logit_model.hessian(beta_iterations[i]))

        # build hat matrix
        rootW = np.sqrt(W)
        H = np.dot(np.transpose(X), np.transpose(rootW))
        H = np.matmul(var_covar_mat, H)
        H = np.matmul(np.dot(rootW, X), H)

        # penalised score
        U = np.matmul(np.transpose(X), y - pi + np.multiply(np.diagonal(H), 0.5 - pi))
        new_beta = beta_iterations[i] + np.matmul(var_covar_mat, U)

        # step halving
        j = 0
        while firth_likelihood(new_beta, logit_model) > firth_likelihood(beta_iterations[i], logit_model):
            new_beta = beta_iterations[i] + 0.5*(new_beta - beta_iterations[i])
            j = j + 1
            if (j > step_limit):
                sys.stderr.write('Firth regression failed\n')
                return None

        beta_iterations.append(new_beta)
        if i > 0 and (np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) < convergence_limit):
            break

    return_fit = None
    if np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) >= convergence_limit:
        sys.stderr.write('Firth regression failed\n')
    else:
        # Calculate stats
        fitll = -firth_likelihood(beta_iterations[-1], logit_model)
        intercept = beta_iterations[-1][0]
        beta = beta_iterations[-1][1:].tolist()
        bse = np.sqrt(np.diagonal(-logit_model.hessian(beta_iterations[-1])))
        
        return_fit = intercept, beta, bse, fitll

    return return_fit

#if __name__ == "__main__":

import sys
import warnings
import math
import statsmodels
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
  
# create X and y here. Make sure X has an intercept term (column of ones)
# ...
X = df[feature_cols] # Features
y = df.OR03_H.copy() # Target variable  # si no haces la copia, se sobreescribirían la variable df originalc con la siguiente linea :/
y[(y==1)|(y==2)|(y==3)]=0
y[(y==4)|(y==5)]=1

#y = df.CR01_H # Target variable
y.value_counts()

# How to call and calculate p-values
start_vec=None
(intercept, beta, bse, fitll) = fit_firth(y, X,start_vec)

# Wald test
waldp = 2 * (1 - stats.norm.cdf(abs(beta[0]/bse[0])))

# LRT
null_X = np.delete(X, 1, axis=1)
(null_intercept, null_beta, null_bse, null_fitll) = fit_firth(y, null_X)
lrstat = -2*(null_fitll - fitll)
lrt_pvalue = 1
if lrstat > 0: # non-convergence
    lrt_pvalue = stats.chi2.sf(lrstat, 1)
    
    
    
# =============================================================================
#--------------Relationship between independent vbles and dependent variable --
#--------------    Density Graphs
# =============================================================================
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('dianhebo', 'QflsICWh3KUuipjBw7BO')
#import plotly 
#plotly.tools.set_credentials_file('dianhebo', 'QflsICWh3KUuipjBw7BO')


def scatter_with_color_dimension_graph(feature, target, layout_labels):
    """
    Scatter with color dimension graph to visualize the density of the
    Given feature with target
    :param feature:
    :param target:
    :param layout_labels:
    :return:
    """
    color_names = ['1','2','3','4','5']
    color_vals = list(range(len(color_names)))
    trace1 = go.Scatter(
        y=feature,
        mode='markers',
        marker=dict(
            #size='16',
            color=target,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Helpfulness')
        )
    )
    layout = go.Layout(
        title=layout_labels[2],
        xaxis=dict(title=layout_labels[0]), yaxis=dict(title=layout_labels[1]))
    data = [trace1]
    fig = Figure(data=data, layout=layout)
    # plot_url = py.plot(fig)
    py.image.save_as(fig, filename=layout_labels[2] + '.png')

   
def create_density_graph(dataset, features_header, target_header):
    """
    Create density graph for each feature with target
    :param dataset:
    :param features_header:
    :param target_header:
    :return:
    """
    for feature_header in features_header:
        #print "Creating density graph for feature:: {} ".format(feature_header)
        layout_headers = ["Case", 'Response for ' + feature_header[5:],
                          feature_header + " & " + target_header + " Density Graph"]
        #scatter_with_color_dimension_graph(dataset[feature_header], dataset[target_header], layout_headers)
        scatter_with_color_dimension_graph(dataset[feature_header], dataset[target_header], layout_headers)

# the most helpful review        
h_data_headers = ['OR03_E01', 'OR03_E02', 'OR03_E03', 'OR03_E04', 'OR03_E05', 'OR03_E06', 'OR03_E07', 'OR03_E08', 'OR03_E09', 'OR03_E10', 'OR03_H']
df1=df.sort_values('OR03_H')
create_density_graph(df1, h_data_headers[0:-1], h_data_headers[-1])
# quick validation
df1[(df1['OR03_H']==4) & (df['OR03_E05']==1)]

# the least helpful review CR01_H
h_data_headers = ['CR01_E01', 'CR01_E02', 'CR01_E03', 'CR01_E04', 'CR01_E05', 'CR01_E06', 'CR01_E07', 'CR01_E08', 'CR01_E09', 'CR01_E10', 'CR01_H']
df1=df.sort_values('CR01_H')
create_density_graph(df1, h_data_headers[0:-1], h_data_headers[-1])


# =============================================================================
#--------------SVM----------------------------------------------
# =============================================================================
# import the class
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score

feature_cols = ['OR03_E01', 'OR03_E02', 'OR03_E03', 'OR03_E04', 'OR03_E05', 'OR03_E06', 'OR03_E07', 'OR03_E08', 'OR03_E09', 'OR03_E10']
X = df[feature_cols] # Features
y = df.OR03_H.copy() # Target variable
ybin = df.OR03_H.copy() # Target variable
y.value_counts()

# Less H case
feature_cols = ['CR01_E01', 'CR01_E02', 'CR01_E03', 'CR01_E04', 'CR01_E05', 'CR01_E06', 'CR01_E07', 'CR01_E08', 'CR01_E09', 'CR01_E10']
X = df[feature_cols] # Features
y = df.CR01_H.copy() 
ybin = df.CR01_H.copy() 

#  test the binary case
ybin[(ybin==1)|(ybin==2)|(ybin==3)]=0 # non H at all, non H, neutral
ybin[(ybin==4)|(ybin==5)]=1 #H, vey H
ybin.value_counts()

X_train,X_test,y_train,y_test=train_test_split(X,ybin,test_size=0.25,random_state=0)

# instantiate the model (using the default parameters)
#clf = svm.SVC(gamma=0.001)
clf = svm.SVC()

# fit the model with data
clf.fit(X_train,y_train)
# ten-fold cross-validation
scores = cross_val_score(clf, X, ybin, cv=5)
scores.mean()
# Predict values
#y_pred=clf.predict(X_test)


# SVM multi-class classification-----------------------
# On the other hand, LinearSVC implements “one-vs-the-rest” multi-class strategy, thus training n_class models. If there are only two classes, only one model is trained:
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
lin_clf = svm.LinearSVC()
lin_clf.fit(X_train,y_train) 
scores = cross_val_score(lin_clf, X, y, cv=5)
scores.mean()
scores.shape


# =============================================================================
#--------------Analysis all reviews (but not merged!)--------------------------------------------
# =============================================================================
def get_AA_value(ID_A_Helpfulness, rate_type):
    return codes[codes.ID_A_Helpfulness==ID_A_Helpfulness][rate_type].values[0]
#get_AA_value('OR03_H','facts_rate')
#get_AA_value('OR03_H','negative_op_rate')
#get_AA_value('OR03_H','positive_op_rate')
coef_models=pd.DataFrame()
results_cols={'LogitReg_mc_woAA','LogitReg_bin_woAA','SVM_woAA',
              'LogitReg_mc_wF','LogitReg_bin_wF','SVM_wF',
              'LogitReg_mc_wO','LogitReg_bin_wO','SVM_wO',
              'LogitReg_mc_wFO','LogitReg_bin_wFO','SVM_wFO',
              '_id_h','_h_mean', '_h_rate'}
results_idx = list()
for i in codes['ID_A_Helpfulness']:
    results_idx.append(i[0:5]+'All')
    for j in reasons_suffix:
        results_idx.append(i[0:5]+'E'+j)
results=pd.DataFrame(columns=results_cols,index=results_idx)
combinations=list(['woAA','wF','wO','wFO']) #woAA: without ArguAna, wF: with facts, wO: with opinions, wFO: with facts and opinions

from sklearn import metrics       
all_cols=list()
for i in codes['ID_A_Helpfulness']:
    feature_cols=list()
    feature_cols.append(i[0:5]+'All')
    all_cols.append(i[0:5]+'All')
    for j in reasons_suffix:
        feature_cols.append(i[0:5]+'E'+j)
    # Add ArguAna features
#    feature_cols.append(i[0:5]+'FR')
#    feature_cols.append(i[0:5]+'NR')
#    feature_cols.append(i[0:5]+'PR')
    single_feature=False
    all_exp=False
    for j in feature_cols:
        if j==i[0:5]+'All':
            X = df[feature_cols[1:]] # all explanations
            results_row_name=i[0:5]+'All'
            all_exp=True
        else:
            X = df[j] # every explanation
            results_row_name=j
            single_feature=True
            all_exp=False
        
        # set special X with facts and opinions
        Xf=pd.DataFrame(X.copy())
        Xf[i[0:5]+'FR'] = get_AA_value(i,'facts_rate')
        
        Xo=pd.DataFrame(X.copy())
        Xo[i[0:5]+'NR'] = get_AA_value(i,'negative_op_rate')
        
        Xfo=pd.DataFrame(X.copy())
        Xfo[i[0:5]+'FR'] = get_AA_value(i,'facts_rate')
        Xfo[i[0:5]+'NR'] = get_AA_value(i,'negative_op_rate')
        
        XwAA=X.copy()
        for k in combinations:
            if k=='wF': X = Xf
            if k=='wO': X = Xo
            if k=='wFO': X = Xfo
            if k=='woAA': X = XwAA
            if k!='woAA': single_feature=False
            #print('k: '+str(k)+', j:'+str(j)+", len: "+str(X.shape))
            # Multiclass analysis------------
            y = df[i].copy()
            # Split
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
            if(single_feature): 
                X_train=np.array(X_train).reshape(-1, 1)
                X_test=np.array(X_test).reshape(-1, 1)
                X=np.array(X.copy()).reshape(-1, 1)
            # LogitReg 
            logreg = LogisticRegression(multi_class='multinomial',solver ='newton-cg')
            logreg.fit(X_train,y_train)
            y_pred=logreg.predict(X_test)
            results['LogitReg_mc_'+k][results_row_name]=metrics.accuracy_score(y_test, y_pred)
            
            # Binary analysis------------
            ybin = df[i].copy()
            ybin[(ybin==1)|(ybin==2)|(ybin==3)]=0 # non H at all, non H, neutral
            ybin[(ybin==4)|(ybin==5)]=1 #H, vey H
            # Split
            X_train,X_test,y_train,y_test=train_test_split(X,ybin,test_size=0.25,random_state=0)
            if(single_feature): 
                X_train=np.array(X_train).reshape(-1, 1)
                X_test=np.array(X_test).reshape(-1, 1)
            # LogitReg 
            logreg = LogisticRegression(multi_class='multinomial',solver ='newton-cg')
            logreg.fit(X_train,y_train)
            y_pred=logreg.predict(X_test)
            if all_exp & (k=='wFO'):
                coef_models=coef_models.append(pd.DataFrame(logreg.coef_, index=[i]))
            results['LogitReg_bin_'+k][results_row_name]=metrics.accuracy_score(y_test, y_pred)
            # SVM 
            clf = svm.SVC()
            clf.fit(X_train,y_train)
            scores = cross_val_score(clf, X, ybin, cv=5)
            results['SVM_'+k][results_row_name]=scores.mean()
            
            results['_id_h'][results_row_name]=i
            results['_h_mean'][results_row_name]=float(h_means[h_means[0]==i][1].values[0])
            results['_h_rate'][results_row_name]=float(h_rates[h_rates[0]==i][1].values[0])
# rename cols and rows
coef_models_aux=coef_models.copy()
coef_models.columns=['E01','E02','E03','E04','E05','E06','E07','E08','E09','E10','FR','NR']

# Plots-----------------------------------
#results_all=results.loc[all_cols,:]    
#fig = plt.figure()
#plt.title('Accuracies')
#results.loc[all_cols,:].plot.bar()

#for i in codes['ID_A_Helpfulness']: # Loop reviews
#    feature_cols=list()
#    feature_cols.append(i[0:5]+'All')
#    all_cols.append(i[0:5]+'All')
#    for j in reasons_suffix:
#        feature_cols.append(i[0:5]+'E'+j)
#    metrics={}
#    for j in feature_cols: # loop all and each explanation
#        #print(j[6:])
#        if j[5:]=='All':
#            metricsLR={} 
#            metricsSVM={}
#            for k in combinations: # loop AA, F,O and OF
#                valueLR=results['LogitReg_bin_'+k][j]
#                valueSVM=results['SVM_'+k][j]
#                metricsLR.update({(j,k):valueLR})
#                metricsSVM.update({(j,k):valueSVM})
#            metrics.update({'LR':metricsLR})
#            metrics.update({'SVM':metricsSVM})
#            metricsDf=pd.DataFrame(metrics)
#            #f, a = plt.subplots(3,1)
#            #metricsDf.xs(j).plot(kind='bar',ax=a[0])
#            plt.title('Accuracies '+str(j))
#            metricsDf.xs(j).plot(kind='bar',title='Accuracies '+str(j), ylim=(0.7,0.9)) 
        
# =============================================================================
#--------------Inferential analysis--------------------------------------------
# =============================================================================


# 1: Statistical diff between H and E ratings, bewteen reviews
# 2: Diff E ratings within the same review
# 3: Diff accuracies between models
# 4: Diff coeficients LogReg binary: Explanations, Facts and Opinions rate (between vbles)
# 5: Diff coeficients LogReg binary: Explanations, Facts and Opinions rate (with respect to 0)
            
alpha = 0.05
a1_st_diff=list()
a1_st_no_diff=list()
a2_st_diff=list()
a2_st_no_diff=list()
a3_st_diff=list()
a3_st_no_diff=list()
a4_st_diff=list()
a4_st_no_diff=list()
a5_st_diff=list()
a5_st_no_diff=list()
# cols: H, En
# Rows: CR0n, per hotel

#set combinations along all 18 reviews----------------------
import itertools
suffix=['H','01','02','03','04','05','06','07','08','09','10']
st_difference=list()
st_no_difference=list()
cases=[1,2,3,4]
models=['SVM_woAA','SVM_wF','SVM_wO','SVM_wFO','LogitReg_bin_woAA','LogitReg_bin_wF','LogitReg_bin_wO','LogitReg_bin_wFO']
vbles=['E01','E02','E03','E04','E05','E06','E07','E08','E09','E10','FR','NR']
for k in cases:
    if k==1:
        for j in suffix:
            if j=='H': 
                combinations=list(itertools.combinations(codes['ID_A_Helpfulness'], 2)) 
            else: 
                combinations=list(itertools.combinations(codes.ID_A_Reasons.apply(lambda x: x[0:6]+j), 2)) 
            for i in combinations:
                sample1=df[i[0]]
                sample2=df[i[1]]
                stat, p = mannwhitneyu(sample1, sample2)
                if p > alpha: a1_st_no_diff.append(i)
                else: a1_st_diff.append(i)
    if k==2:
        cols=codes['ID_A_Reasons']
        #cols_all=cols.copy()
        for l in cols:
            cols_all=pd.Series()
            for j in suffix:
                if j!='H':
                    cols_all=cols_all.append(pd.Series([l[0:5]+'E'+j]))
                else:
                    cols_all=cols_all.append(pd.Series([l[0:5]+'H']))
            combinations=list(itertools.combinations(cols_all, 2)) 
            for i in combinations:
                sample1=df[i[0]]
                sample2=df[i[1]]
                stat, p = mannwhitneyu(sample1, sample2)
                if p > alpha: a2_st_no_diff.append(i)
                else: a2_st_diff.append(i)
                
    if k==3:
        combinations=list(itertools.combinations(models, 2))  
        for i in combinations:
            sample1=results[i[0]]
            sample2=results[i[1]]
            stat, p = mannwhitneyu(sample1, sample2)
            if p > alpha: a3_st_no_diff.append(i)
            else: a3_st_diff.append(i)
            
    if k==4: # coeff, between E's and FR and NR
        combinations=list(itertools.combinations(vbles, 2))
        for i in combinations:
            sample1=coef_models[i[0]]
            sample2=coef_models[i[1]]
            stat, p = ttest_ind(sample1, sample2)
            if p > alpha: a4_st_no_diff.append(i)
            else: a4_st_diff.append(i)
    
    if k==5: # coeff, diff between (E's, FR, NR) and 0
        zero_values=np.zeros(coef_models.shape[0])
        for i in vbles:
            sample1=coef_models[i]
            sample2=zero_values
            stat, p = ttest_ind(sample1, sample2)
            if p > alpha: a5_st_no_diff.append(i)
            else: a5_st_diff.append(i)

# check differences -----------A4
a1_st_diff=pd.DataFrame(a1_st_diff)
a2_st_diff=pd.DataFrame(a2_st_diff)
a3_st_diff=pd.DataFrame(a3_st_diff)
a4_st_diff=pd.DataFrame(a4_st_diff)
#a5_st_diff=pd.DataFrame(a5_st_diff)


def get_mean_coeff(review):
    #print(review+str(np.mean(coef_models[review])))
    #np.mean(coef_models[review])
    return np.mean(coef_models[review])

a4_st_diff['c1']=a4_st_diff[0].apply(lambda x: get_mean_coeff(x))
a4_st_diff['c2']=a4_st_diff[1].apply(lambda x: get_mean_coeff(x))




# =============================================================================
#--------------Analyse Facts and Opinions vbles--------------------------------
# =============================================================================
def get_h_mean(id_rev):
    return float(h_means[h_means[0]==id_rev][1].values[0])
def get_h_rate(id_rev):
    return float(h_rates[h_rates[0]==id_rev][1].values[0])
get_h_mean('CR01_H')
Xaux=codes[['ID_A_Helpfulness','facts_rate','negative_op_rate','positive_op_rate', 'review_length']]
Xaux['h_mean']=Xaux.ID_A_Helpfulness.apply(lambda x:get_h_mean(x))
Xaux['h_rate']=Xaux.ID_A_Helpfulness.apply(lambda x:get_h_rate(x))

# Linear regression H mean
y = Xaux['h_mean'].copy()
X = Xaux[['facts_rate','negative_op_rate','review_length']].copy()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred=linreg.predict(X_test)
r2=metrics.r2_score(y_test, y_pred)
print('R2 LinReg (H mean): '+str(r2))
#f1=metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average=None, sample_weight=None)
#print('F1 LogitReg MC, unbalanced: '+str(f1))

y = Xaux['h_rate'].copy()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred=linreg.predict(X_test)
r2=metrics.r2_score(y_test, y_pred)
print('R2 LinReg (H rate): '+str(r2))
coef_model_lin=linreg.coef_
coef_model_lin_pearson=Xaux[['h_mean', 'h_rate', 'facts_rate','negative_op_rate','review_length']].corr(method='pearson')


# =============================================================================
#--------------Analyse ArguAna values vs H,credibility and convincing---------
# =============================================================================
#--------------Mediator variables analysis: only one variable--------------------------------------------
vbles_dep=['H','E05','E08']
#vbles_dep=['H']
vbles_ind=['facts_rate','negative_op_rate','positive_op_rate','review_length']
#vbles_dep=['E08']
#vbles_ind=['review_length']
df_all = pd.DataFrame(columns=vbles_dep) # H,credibility and convincing for all participants, all reviews
df_aux = pd.DataFrame(columns=vbles_dep)
for i in codes['ID_A_Helpfulness']:
    for j in vbles_dep:
        if j!= 'H_rate':
            df_aux[j]=df[i[0:5]+j].astype(float)
    #df_aux['ID_A_Helpfulness']=i[0:4]
    df_aux['ID_A_Helpfulness']=i
    #df_aux['ID_REV']=i
    df_all=df_all.append(df_aux, ignore_index = True)
#df_all=df_all.astype(float)

X_ArguAna=codes[['ID_A_Helpfulness','facts_rate','negative_op_rate','positive_op_rate', 'review_length']] # ArguAna info
X_ArguAna=X_ArguAna.sort_values(by=['ID_A_Helpfulness'])
#i='H'
#j='facts_rate'
# Run linear regresion for each ind variable-----------
for i in vbles_dep:
    if i == 'H_rate':
        y=h_rates.sort_values(by=[0])[1]
    else:
        y=df_all.groupby('ID_A_Helpfulness')[i].mean().sort_index()
    
    for j in vbles_ind:
        X = X_ArguAna[j].copy()
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=0)
        X_train=np.array(X_train).reshape(-1, 1)
        X_test=np.array(X_test).reshape(-1, 1)
        X=np.array(X).reshape(-1, 1)
        
        linreg = LinearRegression()
        linreg.fit(X_train,y_train)
        y_pred=linreg.predict(X_test)
        #print('Facts Coef:',linreg.coef_)
        fig = plt.figure()
        plt.scatter(X_test, y_test,  color='black')
        plt.plot(X_test, y_pred, color='blue', linewidth=3)
        plt.title('Test '+str(i)+', '+str(j))
        r2=metrics.r2_score(y_test, y_pred)
        print('R2 LinReg '+str(i)+', : '+str(j)+': '+str(r2))
        
        # plot predicted with all set
        fig = plt.figure()
        y_pred=linreg.predict(X)
        plt.scatter(X, y,  color='black')
        plt.plot(X, y_pred, color='blue', linewidth=3)
        plt.title('All '+str(i)+', '+str(j))
        
        #since non-linearity, assess polinomial features
        model = make_pipeline(PolynomialFeatures(2),LinearRegression())   #change here poly features!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # the number of degrees was adjusted after the validation curve assesmsent
        model.fit(X_train,y_train)
        
        # evaluate the model on the second set of data
        y_pred = model.predict(X_test)
        r2=metrics.r2_score(y_test, y_pred) 
        print("R2 polynomial regression:"+str(i)+', : '+str(j)+': '+str(r2))
        
        # validation curve, to assess the adequate degrees for polynomial model
        fig = plt.figure()
        degree = np.arange(0, 20)
        train_score, val_score = validation_curve(model, X, y,'polynomialfeatures__degree', degree, cv=10)
        # cv correspond to the number of folds in cross-validation
        
        plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')
        plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
        plt.legend(loc='best')
        #plt.ylim(0, 1)
        plt.xlabel('degree')
        plt.ylabel('score')
        plt.title('Validation curve '+str(i)+', '+str(j));


#--------------
x=df_all['H']
.value_counts()[0]
x.getIndex()
plt.plot(x)

mu = 4
variance = 1
sigma = math.sqrt(variance)
#x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()



x=x.sort_index()
df_all['H'].median()
x=df_all['H']

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

mu = 0
variance = 1
sigma = math.sqrt(variance)
#x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()