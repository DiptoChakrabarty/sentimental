from flask import Flask, render_template,url_for,request
from sklearn.externals import joblib
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
import string,pickle
from sentiment import vector,tf
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer

def patternremove(text,pattern):
    reg = re.findall(pattern,text)
    for pat in reg:
        text = re.sub(pat,"",text)
    return text

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation ])
    return round(count/(len(text) - text.count(" ")),3)*100


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict",methods=["POST"])
def predict():
    if request.method == "POST":
        print(1)
        file = open("feature.pkl",'rb')
        cv= pickle.load(file)
        print(2)
        msg = request.form["message"]
        data = [msg]
        body_len = pd.DataFrame([len(data)- data.count(" ")])
        print(4)
        vect = pd.DataFrame(cv.transform(data).toarray())
        print(3)
        punct = pd.DataFrame([count_punct(data)])
        total_data = pd.concat([punct,vect],axis=1)

        log = joblib.load("model.pkl")
        pred = log.predict(total_data)
        print(pred)
        return render_template("predict.html",pred=pred)







if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)