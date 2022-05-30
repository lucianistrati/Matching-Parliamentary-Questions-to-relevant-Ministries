from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron
from xgboost import XGBClassifier

import pandas as pd
import numpy as np

import fasttext
import spacy
import csv
import os

SAVING_FOLDER = "data"
nlp = spacy.load("fr_core_news_sm")

def create_df_from_arrays(X, Y, saving_filename):
    dict_df = {"text": [], "label": []}

    for (x, y) in list(zip(X, Y)):
        dict_df['text'].append(x)
        dict_df['label'].append(y)

    df = pd.DataFrame.from_dict(dict_df)

    df.to_csv(saving_filename)


def preprocess(X, preprocessing_method=""):
    if preprocessing_method == "":
        return X
    for i in range(len(X)):
        preprocessed_x = []
        doc = nlp(str(X[i]))
        for token in doc:
            if preprocessing_method == "STOPWORDS_ELIM":
                if token.text.lower() not in fr_stop_words:
                    preprocessed_x.append(token.text)
            elif preprocessing_method == "PART_OF_SPEECH":
                preprocessed_x.append(token.text + "_" + token.dep_)
            elif preprocessing_method == "DEPENDENCY":
                preprocessed_x.append(token.text + "_" + token.dep_)

        X[i] = " ".join(preprocessed_x)

    return np.array(X)


def baseline_models(X_train, y_train, X_test, y_test, subject_feature=False):
    models = [BernoulliNB(), MultinomialNB(), Perceptron(),
              AdaBoostClassifier(), RandomForestClassifier(),
              DecisionTreeClassifier(), XGBClassifier()]
    models_names = ["BernoulliNB()", "MultinomialNB()",
                    "Perceptron()", "AdaBoostClassifier()",
                    "RandomForestClassifier()",
                    "DecisionTreeClassifier()", "XGBClassifier()"]


    if subject_feature:
        create_df_from_arrays(X_train, y_train, "data/bert_train.csv")
        create_df_from_arrays(X_test, y_test, "data/bert_test.csv")
        X_train = preprocess(X_train)
        X_test = preprocess(X_test)
        cv = CountVectorizer()
        X_train = cv.fit_transform(X_train)
        X_test = cv.transform(X_test)


    for (model, model_name) in list(zip(models, models_names)):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("*" * 10)
        print(model_name)
        print("ACC: ", accuracy_score(y_pred, y_test))
        print("F1: ", f1_score(y_pred, y_test, average="weighted"))


def save_data(df, train_size, target_column, content_column):
    for i in range(len(df)):
        df.at[i, target_column] = '__label__' + df.iloc[i][
            target_column].replace(" ","_")

    df = df[[target_column, content_column]]
    # the column order needs to be changed for processing with the FastText
    # model
    df = df.reindex(columns=[target_column, content_column])

    df[:int(train_size * len(df))].to_csv(
        os.path.join(SAVING_FOLDER, "train.txt"),
        index=False,
        sep=' ',
        header=False,
        quoting=csv.QUOTE_NONE,
        quotechar="",
        escapechar=" ")

    df[int(train_size * len(df)):].to_csv(
        os.path.join(SAVING_FOLDER, "test.txt"),
        index=False,
        sep=' ',
        header=False,
        quoting=csv.QUOTE_NONE,
        quotechar="",
        escapechar=" ")


def train_fasttext_model(train_size, test_size):
    model = fasttext.train_supervised(input=os.path.join("data",
                                                         "train.txt"),
                                      autotuneValidationFile="data/test.txt")
    model.save_model(
        "model_news_" +
        str(train_size) +
        "-" +
        str(test_size) +
        ".bin")
    return model


def train_fasttext(df, target_column, content_column):
    model_name = "Fasttext Model"
    train_size = 0.8
    test_size = 0.2
    save_data(df, train_size, target_column, content_column)
    # trains the fast-text model on the first (train_size * 100) % of the data
    model = train_fasttext_model(train_size, test_size)
    # tests the fast-text model accuracy on the last ((train_size -
    # test-size) * 100) % of the data


def fasttext_model():
    try:
        os.mkdir("data")
    except FileExistsError:
        pass
    df = pd.read_csv("data/data.csv")
    train_fasttext(df, "label", "text")