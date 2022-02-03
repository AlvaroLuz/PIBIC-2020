import pandas as pd
import numpy as np

import nltk

from tweets2Vec import tweets2Vec
from classifiers import naiveBayes, MLP, SVM_parameters, RF

path_dict =  [{'method':'Tfidf',
               'name':'Tfidf',
               'path_tweet_set':'data/sarcasmo_dataset.csv'
               }
            #   {'method':'Vectors',
            #    'name':'NILC Glove 1000 dim',
            #    'path_csv':'data/sarcasm/NILC_GLOVE1000.csv',
            #    'embeddings':'input/embeddings/GloVe/glove_s1000.txt',
            #    'path_tweet_set':'data/sarcasmo_dataset.csv'
            #   },
            #   {'method':'Vectors',
            #    'name':'Stanford Glove twitter 200 dim',
            #    'path_csv':'data/sarcasm/STANFORD_TWITTER200.csv',
            #    'embeddings':'input/embeddings/GloVe/glove.twitter.27B.200d.txt',
            #    'path_tweet_set':'data/sarcasmo_dataset.csv'
            #   },
            #   {'method':'Vectors',
            #    'name':'Stanford Glove wikipedia 300 dim',
            #    'path_csv':'data/sarcasm/STANFORD_WIKI300.csv',
            #    'embeddings':'input/embeddings/GloVe/glove.6B.300d.txt',
            #    'path_tweet_set':'data/sarcasmo_dataset.csv'
            #   }           
]

for path_set in path_dict:
    if  path_set['method'] == 'Tfidf':
        corpus_tweets = pd.read_csv(path_set['path_tweet_set'],encoding="utf8",sep='\t')
        labels = corpus_tweets['Label'].unique()

        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(norm=None, stop_words="english",max_df=0.95, min_df=0.007)
        tfidf = vectorizer.fit_transform(corpus_tweets['Tweet text'])

        X = tfidf;
        y = corpus_tweets['Label'];
    else:
        tweets2Vec(path_tweet_set=path_set['path_tweet_set'],path_embeddings=path_set['embeddings'],path_csv=path_set['path_csv'])
        tweet_vectors = pd.read_csv(path_set['path_csv'])
        print(tweet_vectors.shape)
        print(pd.isnull(tweet_vectors).sum().sum())

        #splitting for the training of the classifier 
        X = tweet_vectors.drop('Label', axis=1)
        y = tweet_vectors['Label']
        labels = tweet_vectors['Label']

    naiveBayes(X,y, path_set['method'])
    MLP(X,y)
    RF(X,y)
    SVM_parameters(X,y)
