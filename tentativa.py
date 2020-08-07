import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import gensim
import nltk
import fasttext
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
def plotConfusionMatrix(classifier,X_test,y_test,y_pred,class_names):
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,display_labels=class_names,cmap=plt.cm.Blues)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()
    return ;


def createW2VOrFTEmbeddingsCsvForDataSet(path_tweet_set, path_embeddings, path_csv):
    #importing word embeddings
    from gensim.models import KeyedVectors
    wv = KeyedVectors.load_word2vec_format((path_embeddings), binary=False)
    #loading tweets without usernames, URLs and emojis/emoticons
    corpus_tweets = pd.read_csv(path_tweet_set,encoding="utf8",sep='\t')

    #creating an empty dataframe that will receive the embeddings + labels
    tweet_vectors = pd.DataFrame()

    #loading stopwords
    stopwords = nltk.corpus.stopwords.words('english')

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
                    word_vec = wv[word]
                    temp = temp.append(pd.Series(word_vec), ignore_index= True)
                
                except:
                    pass
        #loading the average of the tweets to a buffer
        tweet_vector = temp.mean()
        #loading the buffer into the main dataframe 
        tweet_vectors= tweet_vectors.append(tweet_vector, ignore_index = True)

    #loading labels into the dataframe and nullifying empty spaces
    tweet_vectors['Label'] = corpus_tweets['Label']
    tweet_vectors = tweet_vectors.dropna()
    tweet_vectors.to_csv(path_csv, index=False)
    
    return;


def createGloVeEmbeddingsCsvForDataSet(path_tweet_set, path_embeddings, path_csv):
    
    embeddings_index = dict()
    f = open(path_embeddings,'r')
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Loaded %s word vectors.' % len(embeddings_index))   

    #loading tweets without usernames, URLs and emojis/emoticons
    corpus_tweets = pd.read_csv(path_tweet_set,encoding="utf8",sep='\t')
    
    #creating an empty dataframe that will receive the embeddings + labels
    tweet_vectors = pd.DataFrame()

    #loading stopwords
    stopwords = nltk.corpus.stopwords.words('english')

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
                    temp = temp.append(pd.Series(word_vec), ignore_index= True)
                
                except:
                    pass
        #loading the average of the tweets to a buffer
        tweet_vector = temp.mean()
        #loading the buffer into the main dataframe 
        tweet_vectors= tweet_vectors.append(tweet_vector, ignore_index = True)

    #loading labels into the dataframe and nullifying empty spaces
    tweet_vectors['Label'] = corpus_tweets['Label']
    tweet_vectors = tweet_vectors.dropna()
    tweet_vectors.to_csv(path_csv, index=False)
    

    return;
def loadFTmodel(path_tweet_set, path_embeddings, path_csv):
    
    ft = fasttext.load_model(path_embeddings)
    
    #loading tweets without usernames, URLs and emojis/emoticons
    corpus_tweets = pd.read_csv(path_tweet_set,encoding="utf8",sep='\t')
    
    #creating an empty dataframe that will receive the embeddings + labels
    tweet_vectors = pd.DataFrame()

    #loading stopwords
    stopwords = nltk.corpus.stopwords.words('english')

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
                    word_vec = ft.get_word_vector(word)
                    temp = temp.append(pd.Series(word_vec), ignore_index= True)
                
                except:
                    pass
        #loading the average of the tweets to a buffer
        tweet_vector = temp.mean()
        #loading the buffer into the main dataframe 
        tweet_vectors= tweet_vectors.append(tweet_vector, ignore_index = True)

    #loading labels into the dataframe and nullifying empty spaces
    tweet_vectors['Label'] = corpus_tweets['Label']
    tweet_vectors = tweet_vectors.dropna()
    tweet_vectors.to_csv(path_csv, index=False)
    
    return;

path_tweet_set = 'data/githubTweetsDataSet.txt'

path_dict = [{'method':'W2V',
              'name':'W2V CBOW 300 dimensoes',
              'embeddings':'input/embeddings/W2V/cbow_s300.txt',
              'path_csv':'data/sarcasm/NILC_W2VCBOW300.csv'
             }, 
             {'method':'W2V',
              'name':'W2V Skip-Gram 300 dimensoes',
              'embeddings':'input/embeddings/W2V/skip_s300.txt',
              'path_csv':'data/sarcasm/NILC_W2VSKIP300.csv'
             },
             {'method':'W2V',
              'name':'W2V CBOW 1000 dimensoes',
              'embeddings':'input/embeddings/W2V/cbow_s1000.txt',
              'path_csv':'data/sarcasm/NILC_W2VCBOW1000.csv'
             }, 
             {'method':'W2V',
              'name':'W2V Skip-Gram 1000 dimensoes',
              'embeddings':'input/embeddings/W2V/skip_s1000.txt',
              'path_csv':'data/sarcasm/NILC_W2VSKIP1000.csv'
             }
]
for path_set in path_dict:
    if   (path_set['method'] == 'GloVe'):
        #createGloVeEmbeddingsCsvForDataSet(path_tweet_set=path_tweet_set,path_embeddings=path_set['embeddings'],path_csv=path_set['path_csv'])
        print(' ')
    elif (path_set['method'] == 'W2V') or (path_set['method'] == 'FT'):
        print()
        createW2VOrFTEmbeddingsCsvForDataSet(path_tweet_set=path_tweet_set,path_embeddings=path_set['embeddings'],path_csv=path_set['path_csv'])
        #loadFTmodel(path_tweet_set=path_tweet_set,path_embeddings=path_set['embeddings'],path_csv=path_set['path_csv'])

    if  path_set['method'] == 'Tfidf':
        corpus_tweets = pd.read_csv(path_tweet_set,encoding="utf8",sep='\t')
        labels = corpus_tweets['Label'].unique()

        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(norm=None, stop_words="english",max_df=0.95, min_df=0.007)
        tfidf = vectorizer.fit_transform(corpus_tweets['Tweet text'])

        X = tfidf;
        y = corpus_tweets['Label'];
    else:
        tweet_vectors = pd.read_csv(path_set['path_csv'])
        print(tweet_vectors.shape)
        print(pd.isnull(tweet_vectors).sum().sum())

        #splitting for the training of the classifier 
        X = tweet_vectors.drop('Label', axis=1)
        y = tweet_vectors['Label']
        labels = tweet_vectors['Label']

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

    # -------------TUNING TABLE:----------------
    #               W2V CBOW 50
    # KERNEL  |        PARAMETERS       | SCORE:
    # rbf    -> C = 1    ; gamma = 1    ;  0.58

    #             W2V SKIP-GRAM 50
    # KERNEL  |        PARAMETERS       | SCORE:
    # rbf    -> C = 1    ; gamma = 1    ;  0.56

    #                GloVe 50
    # KERNEL  |        PARAMETERS       | SCORE:
    # rbf    -> C = 1    ; gamma = 1    ;  0.56
    #GloVe 50 = c 1000 gamma 1e-4
    #             FastText CBOW 50
    # KERNEL  |        PARAMETERS       | SCORE:
    # rbf    -> C = 1    ; gamma = 1e-2;  0.58
    # fast text d1000 = f1 score de 0.64/0.61

    #           FastText Skip-Gram 50
    # KERNEL  |        PARAMETERS       | SCORE:
    # rbf    -> C = 1    ; gamma = 1    ;  0.57

    #-------------------------------------------
    print('Printing classification for: %s'%path_set['name'])
    print()

    from sklearn.svm import SVC
    
    svm_clf = SVC(kernel='rbf',C=10, gamma=1e-2)

    svm_clf.fit(X_train, y_train)
    y_true, y_pred = y_test, svm_clf.predict(X_test)
    
    print(classification_report(y_true, y_pred))
    print("Confusion matrix for SVM: ")
    print(confusion_matrix(y_true, y_pred))

    plotConfusionMatrix(svm_clf,X_test,y_test,y_pred,labels)

    from sklearn.neural_network import MLPClassifier

    mlp_clf = MLPClassifier(hidden_layer_sizes= 12, max_iter= 1100, random_state= 6, solver= 'sgd')
    mlp_clf.fit(X_train, y_train)
    print("Detailed classification report for MLP: ")
    y_true, y_pred = y_test, mlp_clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print("Confusion matrix for MLP: ")
    print(confusion_matrix(y_true, y_pred))

    plotConfusionMatrix(mlp_clf,X_test,y_test,y_pred,labels)

    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import MultinomialNB
    
    if(path_set['method']== 'Tfidf'):
        gnb_clf = MultinomialNB()
    else:
        gnb_clf = GaussianNB()
    
    gnb_clf.fit(X_train, y_train)
    y_true , y_pred = y_test, gnb_clf.predict(X_test)
    print("Detailed classification report for Naive Bayes: ")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix for Naive Bayes: ")
    print(confusion_matrix(y_true, y_pred))

    plotConfusionMatrix(gnb_clf,X_test,y_test,y_pred,labels)

    from sklearn.ensemble import RandomForestClassifier

    rfc_clf = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 
    rfc_clf.fit(X_train, y_train)
    y_pred = rfc_clf.predict(X_test)
    print("Detailed classification report for Randon Forest: ")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix for Randon Forest: ")
    print(confusion_matrix(y_test, y_pred))
    plotConfusionMatrix(rfc_clf,X_test,y_test,y_pred,labels)
# -------------------------SVC PARAMETERS TUNING----------------------------
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.model_selection import GridSearchCV

# param_grid =      [{'kernel': ['rbf'], 
#                      'gamma': [1, 5, 1e-2, 1e-3, 1e-4, 1e-5],
#                      'C': [0.001, 0.10, 0.1, 1, 10, 25, 50, 100, 1000]
#                      },

#                     {'kernel': ['sigmoid'], 
#                      'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                      'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
#                     }#,
# #
# #                    {'kernel': ['linear'], 
# #                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
# #                    }
#                    ]
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv,scoring='f1_macro',n_jobs = -1, verbose = 5)
# grid.fit(X, y)

# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))
# print("Grid scores on development set:")
# print()
# means = grid.cv_results_['mean_test_score']
# stds = grid.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, grid.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
# print()


