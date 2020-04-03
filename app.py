from flask import Flask, render_template,url_for,request

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


@app.route("/predict")
def predict:
    if request.method == "POST":
        msg = request.form["message"]
        data = [msg]
        vect = pd.DataFrame(cv.transform(data).toarray())
        body_len = pd.DataFrame([len(data)- data.count(" ")])
        punct = pd.DataFrame([count_punct(data)])
        total_data = pd.concat([body_len,punct,vect],axis=1)
        pred = log.predict(total_data)
        return render_template("predict.html",pred=pred)







if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)