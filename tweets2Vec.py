import pandas as pd
import numpy as np

import nltk

def tweets2Vec(path_tweet_set, path_embeddings, path_csv):
    embeddings_index = dict()
    f = open(path_embeddings,'r')
    count = 0
    for line in f:
        if count == 0:
            size = line.split()[1]
            count+=1
        else:
            values = line.split(' ')
            word = values[0]
            temp = values[1:]
            #temp.remove('\n')
            coefs = np.asarray((temp), dtype='float32')
            embeddings_index[word] = coefs
    f.close()

    print('Loaded %s word vectors.' % len(embeddings_index))   

    #loading tweets without usernames, URLs and emojis/emoticons
    corpus_tweets = pd.read_csv(path_tweet_set,encoding="utf8",sep='\t')
    
    #creating an empty dataframe that will receive the embeddings + labels
    tweet_vectors = pd.DataFrame()

    #loading stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(norm=None, stop_words="english",max_df=0.95, min_df=0.007)
    tfidf = vectorizer.fit_transform(corpus_tweets['Tweet text'])
    word2tfidf = dict(zip(vectorizer.get_feature_names(),vectorizer.idf_))

    #reading every tweet and converting to non capital letters and removing punctuation
    for tweet in corpus_tweets['Tweet text'].str.lower().str.replace('[^a-z]',' '):
        #creating a buffer dataframe to receive the embeddings
        temp = pd.DataFrame()
        #reading every word splitting them by ' '
        for word in tweet.split(' '):
            #removing stopwords
            if word not in stopwords:
                try:
                    #loading the embeddings of the words to the temporary dataframe
                    word_vec = embeddings_index.get(word)
                    word_vec = np.array(word_vec) * word2tfidf[word]
                    temp = temp.append(pd.Series(word_vec), ignore_index= True)
                
                except:
                    pass
        #loading the average of the tweets to a buffer
        tweet_vector = temp.sum(axis = 0)
        #loading the buffer into the main dataframe 
        tweet_vectors= tweet_vectors.append(tweet_vector, ignore_index = True)

    #loading labels into the dataframe and nullifying empty spaces
    tweet_vectors['Label'] = corpus_tweets['Label']
    tweet_vectors = tweet_vectors.dropna()
    tweet_vectors.to_csv(path_csv, index=False)
    

    return
