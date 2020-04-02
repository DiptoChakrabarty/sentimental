from flask import Flask, render_template,url_for,request
import pandas as pd 
import numpy as np 
from nltk.stem.porter import PorterStemmer
import re,string 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)

def remove_pattern(text,pattern):
    reg = re.findall(pattern,text)
    for pat in reg:
        text = re.sub(pat,"",text)
    return text

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation ])
    return round(count/(len(text) - text.count(" ")),3)*100





if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)