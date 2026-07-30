"""
Resume Screening/Classification: Utility function

Contains the necessary functions for processing the data; 
training, implementing, and evaluating the model; 
along with comparing various models, and creating the confusion matrix.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC

try:
    from nltk.corpus import stopwords
    STOPWORDS= set(stopwords.words("english"))
except Exception:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOPWORDS= set(ENGLISH_STOP_WORDS)





# DATA LOADING

def load_data(filepath= "ResumeDataset.csv"):
    "Load the raw dataset with 'text' and 'label' columns; drop empties."
    df= pd.read_csv(filepath)