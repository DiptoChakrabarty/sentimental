import pandas as pd 
import numpy as np 
import nltk
from nltk.stem.porter import PorterStemmer
import re,string 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score

#Classification Models import 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC




##function defination

def patternremove(text,pattern):
    reg = re.findall(pattern,text)
    for pat in reg:
        text = re.sub(pat,"",text)
    return text

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation ])
    return round(count/(len(text) - text.count(" ")),3)*100

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
vectorizer = CountVectorizer(stop_words="english")
vect = vectorizer.fit_transform(data["tidy_text"])
inputdata = pd.concat([data["length"],data["percentage"],pd.DataFrame(vect.toarray())],axis=1)
print(inputdata.head())

#tf-idf 
tf=TfidfVectorizer(stop_words="english")
tfvect = tf.fit_transform(data["tidy_text"])
tfidfdata = pd.concat([data["length"],data["percentage"],pd.DataFrame(vect.toarray())],axis=1)
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
'''
for mod,clf in model:
    scores = cross_val_score(clf,inputdata,data["label"],scoring="accuracy",cv=5)
    print("Model is {} and Score is {}".format(mod,scores.mean()))
'''
#Hyper tuning
'''
param_grid = {"C" : [0.001,0.01,0.1]}
grid = GridSearchCV(LogisticRegression(),param_grid,cv=5)
grid.fit(inputdata,data["label"])

print(grid.best_estimator_)'''

log = LogisticRegression(C=0.1)
scores = cross_val_score(log,inputdata,data["label"],scoring="accuracy",cv=3)
print("Scores obtained is {}".format(scores))

X_train, X_test, y_train, y_test = train_test_split( inputdata,data["label"], test_size=0.33, random_state=42) 

log.fit(X_train,y_train)
print(log.predict(X_test))


