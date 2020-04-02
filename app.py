from flask import Flask, render_template,url_for,request
import pandas as pd 
import numpy as np 
from nltk.stem.porter import PorterStemmer
import re,string 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def 