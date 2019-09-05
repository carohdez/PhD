# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:38:34 2019

@author: Carolina
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import re
from scipy.stats import ttest_ind



# =============================================================================
# Load data
# =============================================================================

data = pd.read_csv('top200.csv')
# all_tweets = pd.read_csv('twitter_export_tweets_1556616359814.csv')
authors_info = pd.read_csv('authors.csv') 

# all_tweets.iloc[0]
stopwords = pd.read_csv('stopwords.txt')
data.rename(columns={'Author screen name':'author', 'Text':'text' }, inplace=True)
for index, row in data.iterrows():
    if not pd.isnull(row['author']):
        author=row['author']
    else:
        data['author'][index] = author

# all_tweets.columns.tolist()

# drop the original categories labeled in the tweets file, so we can merge tweets and complete authors_info dataframe
data=data.drop(['category', 'who', 'none'], axis=1)

# merge tweets and info of authors
data=pd.merge(data, authors_info, how='inner', on='author', left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)

def rename_role(txt):
    if txt=='Government and politicians':
        return 'Gov & Politicians'
    elif txt=='Social activists and communities':
        return 'Social act & Comm'
    elif txt=='Private individual':
        return 'Priv indiv'
    else:
        return txt
    
data['GroupedCategory'] = data.GroupedCategory.apply(lambda x: rename_role(x))
bf_hashtags=["#BlackFriday", "#CyberMonday", "#BFCM"]
def pre_process(text):
    text=re.sub("http\S+", " ", text) 
    #text=re.sub("(\\d|\\W)+"," ",text) #removes puntuation is also removing emojies
    text=text.lower()
    return text

def remove_for_freq(text):
    text= re.sub("(\\d|\\W)+"," ",text) #removes puntuation is also removing emojies
    for i in bf_hashtags:
        text=re.sub(i+"\S+", " ", text) #removes bf hashtags
    return text

def get_bf_hashtags(text):
    n_bf_hashtags=0
    for i in bf_hashtags:
        n_bf_hashtags=n_bf_hashtags+len([m.start() for m in re.finditer(i, text)])
    return n_bf_hashtags
#data.text=data.text.apply(lambda x: pre_process(x))
data['original_text']=data['text'] # with all elements
data['text']=data.text.apply(lambda x: pre_process(x)) #remove url and lower case
data['text_no_punct']=data.text.apply(lambda x: remove_for_freq(x)) # remove puntuation, and emojis

# add cols
data['length']=data.text.apply(lambda x : len(x)) #length without urls, because limit of 280 excludes urls
data['number_words']=data.text.apply(lambda x : len(x.split()))

data['urls']=data.original_text.apply(lambda x : len(re.findall("http\S+", x)))
data['hashtags']=data.original_text.apply(lambda x : 0 if len(re.findall("#\S+", x))<2 else len(re.findall("#\S+", x))-get_bf_hashtags(x)) # minus the blackfriday hashtag
data['mentions']=data.original_text.apply(lambda x : len(re.findall("@\S+", x)))
data['emojis']=data.original_text.apply(lambda x : len(re.findall(r'[\U0001f300-\U0001f999]', x)))

data['has_urls']=data.urls.apply(lambda x : 1 if x>0 else 0)
data['has_hashtags']=data.hashtags.apply(lambda x : 1 if x>0 else 0)
data['has_mentions']=data.mentions.apply(lambda x : 1 if x>0 else 0)
data['has_emojis']=data.emojis.apply(lambda x : 1 if x>0 else 0)


# Safe csv with processed data 
cols_=data.columns.values
cols_= ['Role','author', 'followers', 'verified (blue checked)',
       'length', 'number_words', 'urls', 'hashtags',
       'mentions', 'emojis', 'has_urls', 'has_hashtags', 'has_mentions',
       'has_emojis', 'first_person_words', 'second_person_words','third_person_words', 'all_person_words', 'polarity']
cols_text= ['Role','author','original_text','followers', 'verified (blue checked)',
       'length', 'number_words', 'urls', 'hashtags',
       'mentions', 'emojis', 'has_urls', 'has_hashtags', 'has_mentions',
       'has_emojis', 'first_person_words', 'second_person_words','third_person_words', 'all_person_words', 'polarity']

#all info all roles
name_file='All_TweetsInfo.csv'
df=data[data.notna().Role]
df.to_csv(name_file,index=False)

# Text, All
name_file="Tweets_AllRoles_WithText.csv"
df=data[data.notna().Role][cols_text]
df.to_csv(name_file,index=False)

# Text, Companies and influencers
name_file="Tweets_CompInfl_WithText.csv"
df=data[(data['Role']=='Companies')|(data['Role']=='Influencers')][cols_text]
df.to_csv(name_file,index=False)

# No Text, All
name_file="Tweets_AllRoles_NoText.csv"
df=data[data.notna().Role][cols_]
df.to_csv(name_file,index=False)

# No Text, Companies and influencers
name_file="Tweets_CompInfl_NoText.csv"
df=data[(data['Role']=='Companies')|(data['Role']=='Influencers')][cols_]
df.to_csv(name_file,index=False)


data=data.rename(columns={'Role':'GroupedCategory'})

# =============================================================================
# Get  authors info from all tweets file
# =============================================================================

#authors=data['author'].unique().tolist()
#def get_author_info(author, col):
#    value=""
#    if (col=='who') | (col=='category'):
#        value=data[data['author']==author][col].tolist()[0]
#    if col=='followers':
#        value=all_tweets[all_tweets['Author screen name']==author]['Author followers count'].unique().tolist()[0]
#    if col=='verified':
#        value=all_tweets[all_tweets['Author screen name']==author]['Author verified'].unique().tolist()[0]     
#    if col=='description':
#        value=all_tweets[all_tweets['Author screen name']==author]['Author description'].unique().tolist()[0]
#    return value
##author='NASA'
##col='followers'
##get_author_info('NASA','followers')
#authors_info = pd.DataFrame({'author':authors})
##authors_info=authors_info.rename(columns={'0':'author'})
##authors_info.add(columns=['who','category','description','followers','verified'])
##
#authors_info["who"]=authors_info.author.apply(lambda x: get_author_info(x,'who'))
#authors_info["category"]=authors_info.author.apply(lambda x: get_author_info(x,'category'))
#authors_info["description"]=authors_info.author.apply(lambda x: get_author_info(x,'description'))
#authors_info["followers"]=authors_info.author.apply(lambda x: get_author_info(x,'followers'))
#authors_info["verified"]=authors_info.author.apply(lambda x: get_author_info(x,'verified'))
#data["type_author"]=data.author.apply(lambda x: get_author_info(x,'category'))
#aux=authors_info["category"].unique().tolist()
#
# =============================================================================
# Get  authors info from csv file
# =============================================================================
##Index(['author', 'who', 'Jeny/Sandra tagging', 'Category', 'Subcategory',
##       'GroupedCategory', 'Topic', 'GroupedTopic', 'description', 'followers',
##       'verified (blue checked)']
##    
#data_aux=data[data['author']=='NASA']
#authors_info['n_tweets']= authors_info.author.apply(lambda x: data[data['author']==x].shape[0])
#authors_info['avg_tweet_lenght']= authors_info.author.apply(lambda x: data[data['author']==x].shape[0])




# =============================================================================
# Metrics 
# =============================================================================

# Frequency of tweets--------------------------------
# Total count of tweets
ax=data.groupby(['GroupedCategory']).count()[['text']].sort_values(by='text', ascending=False).plot.bar(title='Total number of tweets per category per day', rot=60, fontsize=16, legend=False)
ax.set_xlabel("Role", size=18)
ax.set_ylabel("# of tweets", size=18)
ax.set_title('Total number of tweets per category per day', size=22)

# Mean of tweets per category
grp=data.groupby(['author','GroupedCategory'])
err=grp.count()[['text']].groupby(['GroupedCategory']).std()[['text']]
ax=grp.count()[['text']].groupby(['GroupedCategory']).mean()[['text']].sort_values(by='text', ascending=False).plot.bar(title='Mean tweets per author', yerr=err, rot=60, fontsize=16, legend=False)
ax.set_xlabel("Role", size=18)
ax.set_ylabel("Mean of # tweets", size=18)
ax.set_title('Mean tweets per author', size=22)
# Max number of tweets per category
ax=grp.count()[['text']].groupby(['GroupedCategory']).max()[['text']].sort_values(by='text', ascending=False).plot.bar(title='Max tweets per author', rot=60, fontsize=16, legend=False)
ax.set_xlabel("Role", size=18)
ax.set_ylabel("Max of # tweets", size=18)
ax.set_title('Max tweets per author', size=22)
max_num_tweets_role=grp.count()[['text']].groupby(['GroupedCategory']).max()[['text']].text.max()
grp=data.groupby('GroupedCategory')

# Length of tweets--------------------------------
# mean of lenght
ax=data['length']
plot.bar()
ax=grp.mean()[['length']].sort_values(by='length', ascending=False).plot.bar(title='Mean of tweet length',yerr=grp.std()[['length']], rot=60, fontsize=16, legend=False)
ax.set_xlabel("Role", size=18)
ax.set_ylabel("Mean of tweet length", size=18)
ax.set_title('Mean of tweet length', size=22)
# mean number of words
ax=grp.mean()[['number_words']].sort_values(by='number_words', ascending=False).plot.bar(title='Mean of number of words in tweet',yerr=grp.std()[['number_words']], rot=60, fontsize=16, legend=False)
ax.set_xlabel("Role", size=18)
ax.set_ylabel("Mean of # of words in tweet", size=18)
ax.set_title('Mean of number of words in tweet', size=22)

# Padding a df with lenght of tweets per role, to allow creation of boxplot
cols=[ 'Companies', 'Media', 'Gov & Politicians','Social act & Comm','Priv indiv', 'Influencers','Artists', 'Sports team']
# cols=data.GroupedCategory.unique() # use this when you realize how to drop the nan nonstring
df=pd.DataFrame(np.nan, index=np.arange(0,max_num_tweets_role,1), columns=cols)
for i in cols:
    df[i]=pd.Series(data[data['GroupedCategory']==i]['length'].values, index=np.arange(0,data[data['GroupedCategory']==i]['length'].size,1))
# this way we don't get hem ordered
ax=df.plot.box(rot=60, fontsize=16)
ax.set_xlabel("Role", size=18)
ax.set_ylabel("# of characters", size=18)
ax.set_title('Length of tweets', size=22)

# with number of words:
cols=[ 'Companies', 'Media' ,'Social act & Comm', 'Artists', 'Priv indiv','Gov & Politicians', 'Influencers', 'Sports team']
df=pd.DataFrame(np.nan, index=np.arange(0,max_num_tweets_role,1), columns=cols)
for i in cols:
    df[i]=pd.Series(data[data['GroupedCategory']==i]['number_words'].values, index=np.arange(0,data[data['GroupedCategory']==i]['number_words'].size,1))
# this way we don't get hem ordered
ax=df.plot.box(rot=60, fontsize=16)
ax.set_xlabel("Role", size=18)
ax.set_ylabel("# of words", size=18)
ax.set_title('Length of tweets (in words)', size=22)


# Grouped plot % of tweets PAPER PLOT

digits_round=1
df=pd.DataFrame(np.round(grp.sum()[['has_urls']]['has_urls']/grp.count()[['text']]['text']*100,digits_round), columns=['urls'])
df['hashtags']=np.round(grp.sum()[['has_hashtags']]['has_hashtags']/grp.count()[['text']]['text']*100,digits_round)
df['mentions']=np.round(grp.sum()[['has_mentions']]['has_mentions']/grp.count()[['text']]['text']*100,digits_round)
df['emojis']=np.round(grp.sum()[['has_emojis']]['has_emojis']/grp.count()[['text']]['text']*100,digits_round)
df.sort_values(by=df.columns.tolist(), inplace=True, ascending=False)
ax=df.plot.bar(ylim=(0,100), rot=60, fontsize=22)
ax.set_xlabel("Role", size=size_axis_label)
ax.set_ylabel("% of tweets", size=size_axis_label)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005), size=16)
#ax.set_title('% of tweets with elements', size=22)

#T-tests-----------------------------------------------------------------------------
from scipy.stats import mannwhitneyu
variables=['length','number_words','urls','hashtags','mentions','emojis']
cols=['Companies','Influencers']
alpha=[0.05,0.01,0.001]
for i in variables:
    #stat, p = ttest_ind(data[data['GroupedCategory']==cols[0]][i], data[data['GroupedCategory']==cols[1]][i])
    stat, p = mannwhitneyu(data[data['GroupedCategory']==cols[0]][i], data[data['GroupedCategory']==cols[1]][i])
    for j in alpha:
        if p<j:
            print("Distributions of "+i+" of tweets from "+cols[0]+" and "+cols[1] +" are significantly different, p<"+str(j))


# =============================================================================
# Writing style
# =============================================================================
# self-focused
import string
def count_occurrences(text, patterns):
    text=text.lower()
    text=text.translate(str.maketrans('', '', string.punctuation))
    text=re.sub("’", " ", text)
    text=text.split()
    occurrences=0
    for j in patterns:
        occurrences=occurrences+len([i for i, x in enumerate(text) if x == j]) # a comprehension!!
    return occurrences
first_person_words=['i', 'we', 'me', 'us', 'my', 'our', 'mine', 'ours','myself','ourselves']
second_person_words=['you', 'your', 'yours', 'yourself']
third_person_words=['he', 'they', 'him', 'them', 'his', 'her', 'their', 'she', 'her', 'hers', 'theirs', 'it', 'its','itself', 'herself','himself','themselves']
# add column
# data['self_words'] = pd.Series(np.zeros(data.shape[0]), index=data.index)
# data['others_words'] = pd.Series(np.zeros(data.shape[0]), index=data.index)
data['first_person_words'] = data.text.apply(lambda x: count_occurrences(x,first_person_words))
data['second_person_words'] = data.text.apply(lambda x: count_occurrences(x,second_person_words))
data['third_person_words'] = data.text.apply(lambda x: count_occurrences(x,third_person_words))
data['all_person_words'] = data.first_person_words + data.second_person_words + data.third_person_words

grp=data.groupby(['author','GroupedCategory'])

first_words_role=grp.sum().first_person_words.groupby(['GroupedCategory']).sum() # numero de first person words per role
second_words_role=grp.sum().second_person_words.groupby(['GroupedCategory']).sum() # numero de second person words per role
third_words_role=grp.sum().third_person_words.groupby(['GroupedCategory']).sum() # numero de third person words per role
all_persons_words_role=(grp.sum().first_person_words+grp.sum().second_person_words+grp.sum().third_person_words).groupby(['GroupedCategory']).sum() # numero of all person words per role

#pd.DataFrame(first_words_role/all_persons_words_role, columns=['fraction']).sort_values(by= 'fraction', ascending=False).plot.bar(title='Fraction of first person words', ylim=(0,1))
#pd.DataFrame(second_words_role/all_persons_words_role, columns=['fraction']).sort_values(by= 'fraction', ascending=False).plot.bar(title='Fraction of second person words', ylim=(0,1))
#pd.DataFrame(third_words_role/all_persons_words_role, columns=['fraction']).sort_values(by= 'fraction', ascending=False).plot.bar(title='Fraction of third person words', ylim=(0,1))

df=pd.DataFrame(first_words_role.values, index= first_words_role.index, columns=['first_person_words'])
df['second_person_words']=second_words_role.values
df['third_person_words']=third_words_role.values
df=df.T
for i in df.ix[0].index.values:
    plot = df.plot.pie(y=i, figsize=(5, 5), subplots=True, title='Distribution of person words, role '+i)
    
for i in df.index.values:
    y=df.ix[i]/df.sum()
    ax=pd.DataFrame(y, columns=['Role']).sort_values(by='Role', ascending=False).plot.bar(title="% of use of "+i, ylim=[0,1], rot=60, fontsize=16, legend=False)
    ax.set_xlabel("Role", size=18)
    ax.set_ylabel("% of words", size=18)
    ax.set_title("% of use of "+i, size=22)

#T-tests-----------------------------------------------------------------------------
variables=['first_person_words','second_person_words','third_person_words']

#cols=['Companies','Influencers']
cols=['Artists','Gov & Politicians']
alpha=[0.05,0.01,0.001]
for i in variables:
    stat, p = ttest_ind(data[(data['GroupedCategory']==cols[0]) & (data['all_person_words']>0)][i]/data[(data['GroupedCategory']==cols[0]) & (data['all_person_words']>0)]['all_person_words'], data[(data['GroupedCategory']==cols[1]) & (data['all_person_words']>0)][i]/data[(data['GroupedCategory']==cols[1]) & (data['all_person_words']>0)]['all_person_words'])
    for j in alpha:
        if p<j:
            print("Means of "+i+" for "+cols[0]+" and "+cols[1] +" are significantly different, confidence "+str(j))

# Stack plot
df1=pd.DataFrame(df.sum(),columns=['words'])
df=pd.DataFrame(df.T.stack(),columns=['words'])
df=df/df1*100
df=df.unstack()
df.sort_values(by=df.columns.tolist(), inplace=True, ascending=False)
df=df.stack()

ax=df.unstack().plot(kind='bar', stacked=True, rot=60, fontsize=22, ylim=[0,119])
ax.set_xlabel("Role", size=22)
ax.set_ylabel("% of tweets", size=22)
#ax.set_title("Writing style: Use of words in reference to self and others", size=22)
ax.legend(['1st Person','2nd Person', '3rd Person'], fontsize=18, frameon=False, loc='upper center', ncol=3)


# =============================================================================
# Wordcloud
# =============================================================================
import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
% matplotlib inline

# Start with one tweet:
text = data.text_no_punct[0] # text without puntuation, in order to include terms within hashtags

# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# all tweets
categories=data.GroupedCategory.unique()
categories=['Companies', 'Influencers']
for i in categories:
    if not pd.isnull(i):
        text=" ".join(tweet for tweet in data.text[data.GroupedCategory==i])
        text=re.sub("http\S+", " ", text)
        #print ("There are {} words in the combination of all tweets.".format(len(text)))
        print("Category:", i)
        # Create stopword list:
        stopwords = set(STOPWORDS)
        stopwords.update(["BlackFriday", "Black Friday", "Black", "Friday","BlackFriday2018", "rt", "will", "today"])
        
        # Generate a word cloud image
        wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=10).generate(text)
        wordcloud.to_file("img/"+i+"_Wordcloud.png")
        
        # Display the generated image:
        # the matplotlib way:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        
# =============================================================================
# Miscelaneous
# =============================================================================

# Take random list of authors
from random import randint
l = [randint(0, 200) for p in range(0, 200)]
authors_subset=authors_info.ix[l[0:55]]

grp=data.groupby(['GroupedCategory', 'Category', 'Subcategory'])
grp.Category.count()


data[data.GroupedCategory=='Social act & Comm' & (data.)].Category
d=data[['GroupedCategory', 'Category', 'Subcategory']].fillna("none")
grp=d.groupby(['GroupedCategory', 'Category', 'Subcategory'])
s=grp.Category.count()
s.index.levels
data


# =============================================================================
# Sentiment analysis
# =============================================================================
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
    
# test
for index,row in data.iterrows():
    sentence=row['text']
    pol_scores = sid.polarity_scores(sentence)
    print(sentence)
    print(pol_scores)
def get_polarity(tweet):
    sid = SentimentIntensityAnalyzer()
    pol_scores = sid.polarity_scores(tweet)
    if pol_scores['compound'] > 0.05:
        return 'positive'
    elif pol_scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'
    
#    compound score >= 0.05
#    neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
#    negative sentiment: compound score <= -0.05
    #return pd.Series(pol_scores).drop(labels=['compound']).idxmax()        
data['polarity']=data.text.apply(lambda x: get_polarity(x))
positive=data[data['polarity']=='positive']['text']
negative=data[data['polarity']=='negative']['text']

#sentence=data.ix[0]['text']

grp=data.groupby(['GroupedCategory','polarity'])
grp1=data.groupby(['GroupedCategory'])
df=grp.count()[['text']]/grp1.count()[['text']]*100
df=df.unstack()
df.sort_values(by=df.columns.tolist(), inplace=True, ascending=False)
#l.append(('text', 'pos'))
#l.append(('text', 'neg'))
#df.sort_values(by=l, inplace=True, ascending=False)
df=df.stack()
ax=df.unstack().plot(kind='bar', color=['red', 'gray','green'], stacked=True, title='Proportion of polarity per role', rot=60, fontsize=16, ylim=[0,119])
ax.set_xlabel("Role", size=18)
ax.set_ylabel("% of tweets", size=18)
ax.set_title("Polarity of tweets", size=22)
ax.legend(['Negative','Neutral','Positive'], fontsize=14)


# =============================================================================
# TF-IDF + Clustering Analysis 
# =============================================================================

# Pre-process lowercase, drop urls and special chars

def pre_process(text):
    text=text.lower()
    text=re.sub("http\S+", " ", text)
    text=re.sub("(\\d|\\W)+"," ",text)
    return text

#data.text=data.text.apply(lambda x: pre_process(x))
data['original_text']=data['text']
data['text']=data.text.apply(lambda x: pre_process(x))
# TF-IDF preliminary analysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#cv = CountVectorizer(stop_words=list(stopwords))
cv = CountVectorizer(stop_words='english')
word_count_vector= cv.fit_transform(data['text'].tolist())
word_count_vector[0,0]
vocabulary=list(cv.vocabulary_.keys())

tfidf_transformer=TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)


    
# =============================================================================
# Cluster analysis, second approach 
# =============================================================================
        
import nltk
stopwords = nltk.corpus.stopwords.words('english')
# to get stems
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in data['text']:
    allwords_stemmed = tokenize_and_stem(i) #for each tweet, tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
    
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

# Tf-idf and document similarity--------------------
from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

%time tfidf_matrix = tfidf_vectorizer.fit_transform(data['text']) #fit the vectorizer to tweets

print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()


from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

#K-means clustering
from sklearn.cluster import KMeans

num_clusters = 5
km = KMeans(n_clusters=num_clusters)
%time km.fit(tfidf_matrix)
clusters = km.labels_.tolist()


from sklearn.externals import joblib

# to save your model 
joblib.dump(km,  'doc_cluster.pkl')

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()


tweets = { 'author': data['author'], 'text': data['text'], 'cluster': clusters, 'type_author': data['type_author'] }
frame = pd.DataFrame(tweets, index = [clusters] , columns = ['author', 'type_author', 'cluster'])

frame['cluster'].value_counts() #number of tweets per cluster (clusters from 0 to 4)

# TODO: agregate and calculate mean of retweets, ej
grouped = frame['type_author'].groupby(frame['cluster']) #groupby cluster for aggregation purposes
#grouped.mean() #average rank (1 to 100) per cluster

from __future__ import print_function

print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace
    
#    print("Cluster %d titles:" % i, end='')
#    for title in frame.ix[i]['title'].values.tolist():
#        print(' %s,' % title, end='')
    print() #add whitespace
    print() #add whitespace
    
# =============================================================================
# Final paper plots
# =============================================================================
# values for graphs:
s_legend=13
s_label=24
s_ticks=18
s_patches=12


# Length------------------
# Padding a df with lenght of tweets per role, to allow creation of boxplot
cols=[ 'Companies', 'Influencers']
# cols=data.GroupedCategory.unique() # use this when you realize how to drop the nan nonstring
grp=data.groupby(['author','GroupedCategory'])
max_num_tweets_role=grp.count()[['text']].groupby(['GroupedCategory']).max()[['text']].text.max()
grp=data.groupby('GroupedCategory')
df=pd.DataFrame(np.nan, index=np.arange(0,max_num_tweets_role,1), columns=cols)

for i in cols:
    df[i]=pd.Series(data[data['GroupedCategory']==i]['length'].values, index=np.arange(0,data[data['GroupedCategory']==i]['length'].size,1))
# this way we don't get hem ordered
fig = plt.figure(dpi=600)
plt.gcf().subplots_adjust(left=0.25, bottom=0.15)
ax = fig.add_subplot(111)  # create an axes object in the figure
ax=df.plot.box(ax=ax,fontsize=s_ticks, figsize=(5,5), widths=(0.4,0.4))
ax.set_xlabel("Role", size=s_label)
ax.set_ylabel("# of characters", size=s_label)
plt.savefig('PaperImg\Length.png')

# Elements--------------------------
digits_round=1
grp=data[(data['GroupedCategory']==cols[0]) | (data['GroupedCategory']==cols[1])].groupby('GroupedCategory')
df=pd.DataFrame(np.round(grp.sum()[['has_urls']]['has_urls']/grp.count()[['text']]['text']*100,digits_round), columns=['urls'])
df['hashtags']=np.round(grp.sum()[['has_hashtags']]['has_hashtags']/grp.count()[['text']]['text']*100,digits_round)
df['mentions']=np.round(grp.sum()[['has_mentions']]['has_mentions']/grp.count()[['text']]['text']*100,digits_round)
df['emojis']=np.round(grp.sum()[['has_emojis']]['has_emojis']/grp.count()[['text']]['text']*100,digits_round)
#df.sort_values(by=df.columns.tolist(), inplace=True, ascending=False)

fig = plt.figure(dpi=600)
plt.gcf().subplots_adjust(left=0.25, bottom=0.15)
ax = fig.add_subplot(111)  # create an axes object in the figure
ax=df.plot.bar(ax=ax, ylim=(0,119), fontsize=s_ticks, rot=0, figsize=(5,5))
ax.set_xlabel("Role", size=s_label)
ax.set_ylabel("% of tweets", size=s_label)
ax.legend(fontsize=s_legend, frameon=False, loc='upper center', ncol=2)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005), size=s_patches)
plt.savefig('PaperImg\Elements.png')
    
# comm style ---------------------------
grp=data[(data['GroupedCategory']==cols[0]) | (data['GroupedCategory']==cols[1])].groupby(['author','GroupedCategory'])

first_words_role=grp.sum().first_person_words.groupby(['GroupedCategory']).sum() # numero de first person words per role
second_words_role=grp.sum().second_person_words.groupby(['GroupedCategory']).sum() # numero de second person words per role
third_words_role=grp.sum().third_person_words.groupby(['GroupedCategory']).sum() # numero de third person words per role
all_persons_words_role=(grp.sum().first_person_words+grp.sum().second_person_words+grp.sum().third_person_words).groupby(['GroupedCategory']).sum() # numero of all person words per role


df=pd.DataFrame(first_words_role.values, index= first_words_role.index, columns=['first_person_words'])
df['second_person_words']=second_words_role.values
df['third_person_words']=third_words_role.values
df=df.T

df=pd.DataFrame(first_words_role.values, index= first_words_role.index, columns=['first_person_words'])
df['second_person_words']=second_words_role.values
df['third_person_words']=third_words_role.values
df=df.T

# Stack plot
df1=pd.DataFrame(df.sum(),columns=['words'])
df=pd.DataFrame(df.T.stack(),columns=['words'])
df=df/df1*100
df=df.unstack()
#df.sort_values(by=df.columns.tolist(), inplace=True, ascending=False)
df=df.stack()


fig = plt.figure(dpi=600)
plt.gcf().subplots_adjust(left=0.3, bottom=0.15)
ax = fig.add_subplot(111)  # create an axes object in the figure
ax=df.unstack().plot(ax=ax, kind='bar', stacked=True, ylim=(0,135), fontsize=s_ticks, rot=0, figsize=(4.8,5.2), width=0.30)
ax.set_xlabel("Role", size=s_label)
ax.set_ylabel("% of tweets", size=s_label)
#ax.set_title("Writing style: Use of words in reference to self and others", size=22)
ax.legend(['1st person','2nd person', '3rd person'], fontsize=s_legend, frameon=False, loc='upper center', ncol=1)
ax.set_yticklabels(['0','20','40','60','80','100',' '])
plt.savefig('PaperImg\CommStyle.png')

#d=data[['original_text','first_person_words','second_person_words','third_person_words']]

# sentiment----------------------------
grp=data[(data['GroupedCategory']==cols[0]) | (data['GroupedCategory']==cols[1])].groupby(['GroupedCategory','polarity'])
grp1=data.groupby(['GroupedCategory'])
df=grp.count()[['text']]/grp1.count()[['text']]*100
df=df.unstack()
df=df.stack()

# intento
fig = plt.figure(dpi=600)
plt.gcf().subplots_adjust(left=0.3, bottom=0.15)
ax = fig.add_subplot(111)  # create an axes object in the figure
df.unstack().plot(ax=ax, kind='bar', color=['red', 'gray','green'], stacked=True, ylim=(0,135), fontsize=s_ticks, rot=0, figsize=(4.8,5.2), width=0.30)
ax.set_xlabel("Role", size=s_label)
ax.set_ylabel("% of tweets", size=s_label)
ax.legend(['Negative','Neutral','Positive'], fontsize=s_legend, frameon=False, loc='upper center', ncol=1)
ax.set_yticklabels(['0','20','40','60','80','100',' '])
plt.savefig('PaperImg\Sentiment.png')