import pandas as pd 
import numpy as np 
import nltk
from nltk.stem.porter import PorterStemmer
import re,string 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score



## Model creation

data = pd.read_csv("sentiment.csv")