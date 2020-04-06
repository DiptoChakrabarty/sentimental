import pandas as pd 
import numpy as np 
import nltk,pickle
from nltk.stem.porter import PorterStemmer
import re,string 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score,confusion_matrix
from sklearn.externals import joblib

#Classification Models import 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC




##function defination
vocabulary = ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']

def patternremove(text,pattern):
    reg = re.findall(pattern,text)
    for pat in reg:
        text = re.sub(pat,"",text)
    return text

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation ])
    return round(count/(len(text) - text.count(" ")),3)*100

def vector():
    vectorizer = CountVectorizer(stop_words="english",vocabulary=vocabulary)
    return vectorizer

def tf():
    tf= TfidfVectorizer(stop_words="english",vocabulary=vocabulary)
    return tf


## Model creation
data = pd.read_csv("sentiment.csv",sep="\t")
data.columns = ["label","body"]
enc = LabelEncoder()
data["label"]= enc.fit_transform(data["label"])

## clean data

## remove twitter handles
data["notweets"] = np.vectorize(patternremove)(data["body"],"@[\w]*")
##remove punctuations
data["notweets"] = data["notweets"].str.replace("[^a-zA-Z#]"," ")
print(data.head())
##tokenize tweets
tokenized = data["notweets"].apply(lambda x: x.split())
stemmer = PorterStemmer()
tokenized = tokenized.apply(lambda x: [stemmer.stem(word) for word in x])

for i in range(len(tokenized)):
    tokenized[i] = " ".join(tokenized[i])
data["tidy_text"] = tokenized
print(data.head())

data["length"] = data["body"].apply(lambda x:len(x) - x.count(" "))
data["percentage"] = data["body"].apply(lambda x: count_punct(x))

words = " ".join([sent for sent in data["tidy_text"]])

## Count Vectorizer 

vectorizer =vector()
vect = vectorizer.fit_transform(data["tidy_text"])
inputdata = pd.concat([data["length"],data["percentage"],pd.DataFrame(vect.toarray())],axis=1)
print(inputdata.head())

#tf-idf 
tfidf=tf()
tfvect = tfidf.fit_transform(data["tidy_text"])
pickle.dump(tfidf.vocabulary_,open("feature.pkl","wb"))
tfidfdata = pd.concat([data["percentage"],pd.DataFrame(vect.toarray())],axis=1)
print(tfidfdata.head())


#models creation
model =[]
model.append(("lr",LogisticRegression()))
model.append(("rf",RandomForestClassifier()))
model.append(("db",DecisionTreeClassifier()))
model.append(("svc",SVC()))
model.append(("knn",KNeighborsClassifier()))

#Cross Validation
#Count Vectorizer and tfidf  change data 

for mod,clf in model:
    scores = cross_val_score(clf,tfidfdata,data["label"],scoring="accuracy",cv=5)
    print("Model is {} and Score is {}".format(mod,scores.mean()))

#Hyper tuning


'''param_grid = {"C" : [0.001,0.01,0.1,1,10]}
grid = GridSearchCV( LogisticRegression(),param_grid,cv=5)
grid.fit(tfidfdata,data["label"])

print(grid.best_estimator_)'''



X_train, X_test, y_train, y_test = train_test_split( tfidfdata,data["label"], test_size=0.20, random_state=42) 
log = LogisticRegression(C=0.001, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

log.fit(X_train,y_train)
pred =log.predict(X_test)
print(accuracy_score(pred,y_test))
print(confusion_matrix(pred,y_test))


'''param_grid = {"n_estimators" : [5,10,30,100]}
grid = GridSearchCV( RandomForestClassifier(),param_grid,cv=5)
grid.fit(inputdata,data["label"])

print(grid.best_estimator_)
forest = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=5,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

forest.fit(X_train,y_train)
pred =forest.predict(X_test)
print(accuracy_score(pred,y_test))
print(confusion_matrix(pred,y_test))'''



joblib.dump(log,"model.pkl")


