import pandas as pd 
import numpy as np 
import nltk
from nltk.stem.porter import PorterStemmer
import re,string 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score



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
#print(data.head())
enc = LabelEncoder()
data["label"]= enc.fit_transform(data["label"])
#print(data.head())

## clean data

## remove twitter handles
data["notweets"] = np.vectorize(patternremove)(data["body"],"@[\w]*")
print(data.head())

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
