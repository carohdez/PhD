#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 11:46:56 2019

@author: caro
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

# =============================================================================
# Analysis  for user study
# =============================================================================
#load csv files
sys.path.append("C:\Python\ArguAna")
path = 'Data/San_Francisco_California'
files = list()

for filename in os.listdir(path):
    files.append(filename)


#file_name = files[1]
#file_name = 'hotel_80780_401_390.xmi'
    
from xml.dom import minidom
review_values = pd.DataFrame(columns=['hotel_id','review_id','stars','sentiment','n_positive','n_negative','n_facts','review_text','review_text_ann'])
segments_ann = pd.DataFrame(columns=['hotel_id','review_id', 'type_ann','ini', 'end', 'statement','id_statement'])
files = files[1:]

#--------------------------
for file_name in files:
    xmldoc = minidom.parse(path+'/'+file_name)

    #Get values from review
    hotel = xmldoc.getElementsByTagName('arguana:HotelData')
    hotel_id = hotel[0].attributes['hotelID'].value
    s=file_name.split('_')[3]
    review_id = s[0:s.index('.')]
    #stars = hotel[0].attributes['stars'].value
    review_text = xmldoc.getElementsByTagName('cas:Sofa')
    review_text = review_text[0].attributes['sofaString'].value
    review_length = len(review_text)
    n_positive=n_negative=n_facts=ini=end=0
    type_ann = 'negative'
    ann_id=statement=''
    opinions = xmldoc.getElementsByTagName('discourse:Opinion')
    id_statement=0
    for i in opinions:
        polarity = i.attributes['polarity'].value  
        if polarity == 'positive': 
            n_positive = n_positive+1
            type_ann = 'positive'
        else: 
            n_negative = n_negative + 1
            type_ann = 'negative'
        ini = int(i.attributes['begin'].value)
        end = int(i.attributes['end'].value)
        ann_id = i.attributes['xmi:id'].value
        statement = review_text[ini:end]
        id_statement=id_statement+1
        segments_ann = segments_ann.append(pd.DataFrame({'hotel_id':[hotel_id],'review_id':[review_id],'type_ann':[type_ann],'ini':[ini],'end':[end], 'statement':[statement], 'id_statement':id_statement}))
    n_opinions =  len(opinions)      
    
    facts = xmldoc.getElementsByTagName('discourse:Fact')
    n_facts = len(facts)
    for i in facts:
        ini = int(i.attributes['begin'].value)
        end = int(i.attributes['end'].value)
        type_ann='fact'
        ann_id = i.attributes['xmi:id'].value
        statement = review_text[ini:end]
        segments_ann = segments_ann.append(pd.DataFrame({'hotel_id':[hotel_id],'review_id':[review_id],'type_ann':[type_ann],'ini':[ini],'end':[end], 'statement':[statement], 'id_statement':id_statement}))
    
    sentiment = xmldoc.getElementsByTagName('category:Sentiment')
    sentiment = sentiment[0].attributes['score'].value
    
    # add review text with annotation within
    segments_aux=segments_ann[segments_ann['review_id']==review_id].sort_values(by=['ini'])
    review_text_ann='hotel_id: '+hotel_id+', review_id:'+review_id + '\n'
    for index_d, row_d in segments_aux.iterrows():
        review_text_ann = review_text_ann+"["+row_d['type_ann']+"_"+str(row_d['id_statement'])+"->] "+row_d['statement']+" [<-"+row_d['type_ann']+"]" 
    review_text_ann = review_text_ann + "\n\n Original review text: " + review_text
    review_text_ann = review_text_ann + "\n _________________________________________________"
    
    review_values = review_values.append(pd.DataFrame({'hotel_id':[hotel_id],'review_id':[review_id], 'review_length':[review_length], 'sentiment':[sentiment]
    , 'n_positive':[n_positive], 'n_negative':[n_negative], 'n_facts':[n_facts], 'review_text':[review_text], 'review_text_ann':[review_text_ann]}),ignore_index=True)
    
    # Get values of annotations
    opinions = xmldoc.getElementsByTagName('discourse:Opinion')
    polarity = opinions[0].attributes['polarity'].value
    

import matplotlib.pyplot as plt

#
i= '224948'
# =============================================================================
# Plots
# =============================================================================
# Plot pros&cons ratios per hotel
i='80780'
for i in pd.unique(review_values['hotel_id']):
    hotel_df = review_values[review_values['hotel_id']==i]    
    hotel_df = hotel_df.sort_values('review_length')
    
#    # Plot Ratio negative reviews per hotel
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    #ax.set_ylim(1500, 5200)
#    ax.plot(hotel_df['review_length'],hotel_df['n_negative']/(hotel_df['n_negative']+hotel_df['n_positive']), color='lightblue', linewidth=3)
#    ax.set_title('Ratio negative reviews hotel '+i)
#    
#    
    # get distribution of reviews per polarity
    sentiment_count = pd.Series([len(hotel_df[hotel_df['sentiment']=='1.0']),len(hotel_df[hotel_df['sentiment']=='2.0'])
        ,len(hotel_df[hotel_df['sentiment']=='3.0']),len(hotel_df[hotel_df['sentiment']=='4.0'])
        ,len(hotel_df[hotel_df['sentiment']=='5.0'])]
        , index=['1','2','3','4','5'])
    
    fig = plt.figure()
    width = 0.35       # the width of the bars: can also be len(x) sequence
    ax = fig.add_subplot(111)
    ind = ['1','2','3','4','5']
    p1 = plt.bar(ind, sentiment_count, width)  
    plt.ylabel('Reviews')
    plt.xlabel('Sentiment score')
    plt.title('Sentiment scores hotel: '+i)
    plt.show() 
    
    # Plot cummulative statements per review per hotel
    N = len(hotel_df)
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence
    
    # just polarity
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p1 = plt.bar(ind, hotel_df['n_negative'], width, color='red')
    p2 = plt.bar(ind, hotel_df['n_positive'], width, color='green',
                 bottom=hotel_df['n_negative'])
    
    plt.ylabel('Statements')
    plt.xlabel('Length of review')
    plt.title('Statements by polarity hotel: '+i)
    plt.legend((p1[0], p2[0]), ('Negative', 'Positive'))
    
    plt.show()
    
    # opinions and facts
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    p1 = plt.bar(ind, hotel_df['n_negative']+hotel_df['n_positive'], width, color='blue')
    p2 = plt.bar(ind, hotel_df['n_facts'], width, color='gray',
                 bottom=hotel_df['n_negative']+hotel_df['n_positive'])
    
    plt.ylabel('Statements')
    plt.xlabel('Length of review')
    plt.title('Statements by facts/opinions hotel: '+i)
    plt.legend((p1[0], p2[0]), ('Opinions', 'Facts'))
    plt.show()
    
    
# Numbers
# 80780
# 224948
fig=plt.figure()
plt.plot(review_values[review_values['hotel_id']=='224948']['review_length'])

# get segments given hotel and review id
i='224948'
j='752'
s=segments_ann[(segments_ann['hotel_id'] == i) & (segments_ann['review_id']==j)].sort_values(by=['ini'])
# =============================================================================
# Validate annotations
# =============================================================================
# Validate sections of review not annotated
for i in pd.unique(review_values['hotel_id']):
    for j in pd.unique(review_values[review_values['hotel_id']==i]['review_id']):
        review_length = review_values[(review_values['hotel_id']==i) & (review_values['review_id']==j)]['review_length']
        #reviews = hotel[hotel['review_id'] == j]
        segments = segments_ann[(segments_ann['hotel_id']==i) & (segments_ann['review_id']==j)]
        segments = segments.sort_values(['ini'])
        end=0
        for index, row in segments.iterrows():
            if row['ini'] != 0:
                if (row['ini']-end)>7:
                    print('warning, hotel:',i,', review ', j, ' has sentences without annotation!!', row['ini']-end)
            end = row['end']

        
# Validate facts

help(segments_ann.sort_values)
segments_ann=segments_ann.sort_values(by=['hotel_id','review_id', 'ini'])

# =============================================================================
# Create data structure to export excel and rate by Caro and Sandra
# =============================================================================
# validate length
len(review_values[review_values['hotel_id']=='224948'])


i= '268533'
r_aux = review_values[review_values['hotel_id']==i]
reasons=list(['Helpful?'
,'The review was too short / too long.'
,'The level of detail provided was too little / too much.'
,'The review sounded objective.'
,'The review provided a balanced view of pros and cons.'
,'The review presents effective arguments for the authors point of view.'
,'The review addresses the aspects that are relevant for my purposes'
,'The review has a  stringent flow of arguments.'
,'The review includes proper vocabulary, and does not include spelling or grammar errors.'
,'The review seems credible.'
,'The review contains emotional content.'
,'Review includes comparisons between similar hotels and this one.'
,'Review contains information that might be only episodical.'
])
len(r_aux)  
#r_excel = r_excel.sort_values('')
#
r_excel = pd.DataFrame(columns=['hotel_id','review_id', 'review_length', 'sentiment', 'n_positive', 'n_negative', 'n_facts', 'review_text'])

for index, row in r_aux.iterrows():
    r_excel=r_excel.append(pd.DataFrame([row.values], columns=np.array(row.index)))
    r_excel=r_excel.append(pd.DataFrame({'review_text':reasons}, columns= ['review_text']))
    
    
# =============================================================================
# Extract randomly the hotel and reviews for user study
# =============================================================================
import random
#print(random.choice(pd.unique(review_values['hotel_id'])))
hotel_id = random.choice(pd.unique(review_values['hotel_id']))
review_ids = pd.unique(review_values[review_values['hotel_id']==hotel_id]['review_id'])
r_excel = r_aux = pd.DataFrame(columns=['hotel_id','review_id', 'review_length', 'sentiment', 'n_positive', 'n_negative', 'n_facts', 'review_text'])

# chosen: 
for i in range(1,30):
    rev_id=random.choice(review_ids)
    if not any(r_aux['review_id']==rev_id):
        r_aux = r_aux.append(review_values[(review_values['hotel_id']==hotel_id) & (review_values['review_id']==rev_id)])
    
reasons=list([
'How helpful was this review?'
,'The review was too short / too long.'
,'The level of detail provided was too little / too much.'
,'The review includes an adequate amount of objective statements based on facts.'
,'The review provided a balanced view of pros and cons.'
,'The review provided convincing reasons.'
,'The review addresses the aspects that are relevant for my purposes.'
,'The review has a  stringent flow of arguments.'
,'The review seems credible.'
,'The review contains emotional content.'
,'The review contains information that might be only episodical.'
])
len(r_aux)  
for index, row in r_aux.iterrows():
    r_excel=r_excel.append(pd.DataFrame([row.values], columns=np.array(row.index)))
    r_excel=r_excel.append(pd.DataFrame({'review_text':reasons}, columns= ['review_text']))

sum(r_aux['review_length'])

r=review_values[(review_values['hotel_id']=='112307') | (review_values['hotel_id']=='119658') | (review_values['hotel_id']=='224948')]



# =============================================================================
# Sentiment Analysis all cities
# =============================================================================
#load xmi files ------------------------------------------------------------
sys.path.append("C:\Python\ArguAna")
files = list()

#from pathlib import Path
#for file in Path('Data').glob('**/*.xmi'):
#    print(file)

path = 'Data/'
files = list()
folders = list()
#for folder in os.listdir(path):
#    for file in os.listdir(path+folder):
#        files.append(file)
from pathlib import Path
for file in Path('Data').glob('**/*.xmi'):
    filename=""
    for i in file.parts:
        filename=filename+i+'/'
    files.append(filename[:-1])
        

from xml.dom import minidom
statements = pd.DataFrame(columns=['polarity','statement'])
#files = files[1:]
#--------------------------
for file_name in files:
    xmldoc = minidom.parse(file_name)

    #Get values from review
    hotel = xmldoc.getElementsByTagName('arguana:HotelData')
    review_text = xmldoc.getElementsByTagName('cas:Sofa')
    review_text = review_text[0].attributes['sofaString'].value
    review_length = len(review_text)
    ann_id=statement=''
    opinions = xmldoc.getElementsByTagName('arguana:Opinion')
    for i in opinions:
        polarity = i.attributes['polarity'].value
        ini = int(i.attributes['begin'].value)
        end = int(i.attributes['end'].value)
        statement = review_text[ini:end]
        statements = statements.append(pd.DataFrame({'polarity':[polarity],'statement':[statement]}))
    
    facts = xmldoc.getElementsByTagName('arguana:Fact')
    n_facts = len(facts)
    for i in facts:
        ini = int(i.attributes['begin'].value)
        end = int(i.attributes['end'].value)
        polarity='neutral'
        statement = review_text[ini:end]
        statements = statements.append(pd.DataFrame({'polarity':[polarity],'statement':[statement]}))
        
statements.to_csv('statements_polarity.csv')


# Sentiment analysis ------------------------------------------------------------
# Option 1: One hot encoding + linear regression
# Vectorize: one hot encoding
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(binary=True)
cv.fit(statements.statement)
X = cv.transform(statements.statement)


# Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test=train_test_split(X,statements.polarity,test_size=0.05,random_state=0)

# c: parameter for regularization
for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_test, lr.predict(X_test))))

# c= 0.5 brings the best accuracy
lr = LogisticRegression(C=0.25)
lr.fit(X_train, y_train)
y_pred=lr.predict(X_test)
accuracy_score(y_test, y_pred)

# Final Accuracy: 0.7645!!

lr.classes_ 
# The most discriminating words:
feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), lr.coef_[0]
    )
}
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:5]:
    print (best_negative)

    
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:5]:
    print (best_positive)

# Print confusion matrix 
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
import seaborn as sns

y_test.value_counts()
sum(cnf_matrix[2,:])
sum(sum(cnf_matrix))
sum(y_test.value_counts())

class_names=['negative','neutral','positive'] # name  of classes
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

# remove stopwords
from nltk.corpus import stopwords

english_stop_words = stopwords.words('english')
def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words

no_stop_words = remove_stop_words(statements.statement)

# Normalization - stemming words
def get_stemmed_text(corpus):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

stemmed_statements = get_stemmed_text(no_stop_words)

def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

lemmatized_statements = get_lemmatized_text(no_stop_words)

# n-gramm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(lemmatized_statements)
X = ngram_vectorizer.transform(lemmatized_statements)
#X_test = ngram_vectorizer.transform(reviews_test_clean)

X_train, X_test, y_train, y_test = train_test_split(X, statements.polarity, test_size = 0.05)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_test, lr.predict(X_test))))
    
#Accuracy for C=0.01: 0.6980645161290323
#Accuracy for C=0.05: 0.7470967741935484
#Accuracy for C=0.25: 0.7664516129032258
#Accuracy for C=0.5: 0.7638709677419355
#Accuracy for C=1: 0.7664516129032258
final_ngram = LogisticRegression(C=0.25)
final_ngram.fit(X_train, y_train)
y_pred=final_ngram.predict(X_test)
print ("Final Accuracy: %s" 
       % accuracy_score(y_test, final_ngram.predict(X_test)))

# Final Accuracy: 0.7664

# word counts to maximize power-----------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

wc_vectorizer = CountVectorizer(binary=False)
wc_vectorizer.fit(statements.statement)
X = wc_vectorizer.transform(statements.statement)

X_train, X_test, y_train, y_test = train_test_split(X, statements.polarity, test_size = 0.05)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_test, lr.predict(X_test))))
    
#Accuracy for C=0.01: 0.687741935483871
#Accuracy for C=0.05: 0.7316129032258064
#Accuracy for C=0.25: 0.7425806451612903
#Accuracy for C=0.5: 0.743225806451613
#Accuracy for C=1: 0.7438709677419355
    
final_wc = LogisticRegression(C=1)
final_wc.fit(X_train, y_train)
y_pred=final_wc.predict(X_test)
print ("Final Accuracy: %s" 
       % accuracy_score(y_test, y_pred))

# Final Accuracy: 0.743870967741935


# SVM----------------------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(statements.statement)
X = ngram_vectorizer.transform(statements.statement)
X_train, X_test, y_train, y_test = train_test_split(X, statements.polarity, test_size = 0.05)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    svm = LinearSVC(C=c)
    svm.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_test, svm.predict(X_test))))
    
#Accuracy for C=0.01: 0.76
#Accuracy for C=0.05: 0.7741935483870968
#Accuracy for C=0.25: 0.7690322580645161
#Accuracy for C=0.5: 0.7670967741935484
#Accuracy for C=1: 0.7593548387096775
    
final_svm_ngram = LinearSVC(C=0.05)
final_svm_ngram.fit(X_train, y_train)
print ("Final Accuracy: %s" 
       % accuracy_score(y_test, final_svm_ngram.predict(X_test)))

# Final Accuracy: 0.7741935483870968


# combination-----------------------------------------------------------
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(statements.statement)
X = ngram_vectorizer.transform(statements.statement)

X_train, X_test, y_train, y_test = train_test_split(X, statements.polarity, test_size = 0.05)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    svm = LinearSVC(C=c)
    svm.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_test, svm.predict(X_test))))
    
#Accuracy for C=0.01: 0.76
#Accuracy for C=0.05: 0.7741935483870968
#Accuracy for C=0.25: 0.7690322580645161
#Accuracy for C=0.5: 0.7670967741935484
#Accuracy for C=1: 0.7593548387096775
    
final_svm_ngram = LinearSVC(C=0.05)
final_svm_ngram.fit(X_train, y_train)
print ("Final Accuracy: %s" 
       % accuracy_score(y_test, final_svm_ngram.predict(X_test)))

# Deep neural approach----------------------------------------------
import re

words_in_sentences=np.array('')
max_size_sentence = 0
for i in statements.statement:
    sentence=i.lower()
    words_in_sentences=np.append(words_in_sentences,re.split('\W+', sentence))
    max_size_sentence = max(max_size_sentence,len(re.split('\W+', sentence)))
words_in_sentences = np.unique(words_in_sentences)
df=pd.read_csv('Data/glove.6B/'+'glove.6B.100d.txt', delimiter=' ', header=None)
word_s_emb=df[df[0].isin(words_in_sentences)] #embeddings of the words contained in all the sentences, this is done so the algorithm runs faster

# Get the 3d-array of the words included in the sentences. Shape: (# of sentences, # of words, # dim per word)
dim = word_s_emb.shape[1]-1 # drop the dimension 0, which is the actual string word
sentences_emb = np.zeros((len(statements.statement),max_size_sentence,dim))
























                  
