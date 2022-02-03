from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
    
def naiveBayes(X,y,method):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import MultinomialNB

    params = {}
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.36, random_state=42)
    if method == "Tfidf":
        grid = GridSearchCV(MultinomialNB(), param_grid=params, cv=cv,scoring='f1_macro',n_jobs = -1)
    else:
        grid = GridSearchCV(GaussianNB(), param_grid=params, cv=cv,scoring='f1_macro',n_jobs = -1)
    grid.fit(X, y)

    print("The best parameters for Naive Bayes are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_))
def MLP(X,y):
    from sklearn.neural_network import MLPClassifier
    
    parameter_space = {}

    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.36, random_state=42)
    mlp_gs = MLPClassifier(max_iter= 2000, activation = 'tanh', alpha =  0.0001, hidden_layer_sizes= ((100,)), learning_rate= 'adaptive', solver= 'sgd')
    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=cv, scoring='f1_macro')
    clf.fit(X, y) # X is train samples and y is the corresponding labels
    print("The best parameters for MLP are %s with a score of %0.2f"
        % (clf.best_params_, clf.best_score_))

def RF(X,y):
    from sklearn.ensemble import RandomForestClassifier
    
    parameter_space = {}    
    
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.36, random_state=42)
    rfc_gs = RandomForestClassifier(n_jobs=-1,oob_score = True,max_depth= 30, min_samples_leaf= 10, min_samples_split= 15, n_estimators= 300)
    clf = GridSearchCV(rfc_gs, parameter_space, n_jobs=-1, cv=cv, scoring='f1_macro')
    clf.fit(X, y)     
    print("The best parameters for RF are %s with a score of %0.2f"
        % (clf.best_params_, clf.best_score_))
def SVM(X,y):
    from sklearn.svm import SVC
    
    param_grid = {}
    #w2v
    #c = 1 gamma= 0.001 kernel= rbf
    #ft
    #c= 10 gamma= 1e-5
    #glove
    #C= 10 gamma= 0.0001
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.36, random_state=42)
    grid = GridSearchCV(SVC(gamma = 0.0001, C=10, kernel="rbf"), param_grid=param_grid, cv=cv,scoring='f1_macro',n_jobs = -1)
    grid.fit(X, y)

    print("The best parameters for SVC are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_))
def SVM_parameters(X,y):
    from sklearn.svm import SVC
    
    param_grid =      [{'kernel': ['rbf'], 
                        'gamma': [1, 5, 1e-2, 1e-3, 1e-4, 1e-5],
                        'C': [0.001, 0.10, 0.1, 1, 10, 25, 50, 100, 1000]
                        }
    ] 

    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.36, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv,scoring='f1_macro',n_jobs = -1)
    grid.fit(X, y)

    print("The best parameters for SVC are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_))


def MLP_parameters(X,y):
    from sklearn.neural_network import MLPClassifier
    
    parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive']
    }
    
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.36, random_state=42)
    mlp_gs = MLPClassifier(max_iter= 2000)
    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=cv, scoring='f1_macro')
    clf.fit(X, y) # X is train samples and y is the corresponding labels
    print("The best parameters for MLP are %s with a score of %0.2f"
        % (clf.best_params_, clf.best_score_))

def RF_parameters(X,y):
    from sklearn.ensemble import RandomForestClassifier
    
    parameter_space = {
    "n_estimators" : [100, 300, 500, 800, 1200],
    "max_depth" : [5, 8, 15, 25, 30],
    "min_samples_split" : [2, 5, 10, 15, 100],
    "min_samples_leaf" : [1, 2, 5, 10]
    }    
    
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.36, random_state=42)
    rfc_gs = RandomForestClassifier(n_jobs=-1,oob_score = True)
    clf = GridSearchCV(rfc_gs, parameter_space, n_jobs=-1, cv=cv, scoring='f1_macro')
    clf.fit(X, y)     
    print("The best parameters for RF are %s with a score of %0.2f"
        % (clf.best_params_, clf.best_score_))