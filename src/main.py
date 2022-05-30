from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from models import baseline_models, fasttext_model
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

import json
import os


most_frequent_ministries = {"Ministre de la Santé": 2506,
                            "Ministre de l Environnement": 1541,
                            "Ministre du Développement durable et des "
                            "Infrastructures": 1533,
                            "Ministre de la Justice": 1495,
                            "Ministre des Finances": 1472,
                            "Ministre de l'Education nationale": 1251,
                            "Ministre de la Sécurité sociale": 1078,
                            "Ministre de l'Intérieur": 952,
                            "Premier Ministre": 926}
print(most_frequent_ministries.keys())

def read_questions_file():
    with open("data/questions.json") as json_file:
        questions = json.load(json_file)
    return questions


def main():
    global most_frequent_ministries

    try:
        os.mkdir("data")
    except FileExistsError:
        pass

    print(most_frequent_ministries.keys())

    questions = read_questions_file()
    question_fields_set = set()
    destinations_lengths_list = list()
    df = dict()
    dates_list = []
    ministries_list = []
    analyze_data = True
    use_just_subject_feature = True
    replace_categorical_values = False
    test_size = 0.2

    qp_type_dict = {"Question écrite": 0, "Question urgente": 1,
                    "Question orale": 2, "Question élargie/Question avec "
                                        "débat": 3}

    answer_type_dict = {"oral": 0, "written": 1, "withdrawal": 2, "other": 3,
                        'transformed': 4}

    labels = []
    datapoints_list = []
    documents_list = []

    for i, question_id in enumerate(questions.keys()):
        question_fields_set.add(tuple(questions[question_id].keys()))
        question_destinations_list = questions[question_id]['destinations']

        label = ""

        for question_destination in question_destinations_list:
            if question_destination['ministry'] in \
                    most_frequent_ministries.keys():
                label = question_destination['ministry']
                break

        if len(label):
            if use_just_subject_feature:
                labels.append(label)
                datapoints_list.append(questions[question_id]['subject'])
            else:
                if replace_categorical_values:
                    qp_type_feature = qp_type_dict[questions[question_id][
                        'qp_type']]

                    if questions[question_id]['answer_type'] is None:
                        continue
                    answer_type_feature = answer_type_dict[questions[
                        question_id]['answer_type']]
                    has_answer_feature = int(questions[question_id][
                                                 'has_answer'])
                    datapoints_list.append([qp_type_feature,
                                            answer_type_feature,
                                            has_answer_feature])
                labels.append(label)
                documents_list.append(questions[question_id]['subject'])

        if analyze_data:
            for field_name in questions[question_id].keys():
                if field_name == "authors":
                    authors = questions[question_id]["authors"]
                    questions[question_id][field_name] = " ".join(authors)

                if field_name == "destinations":
                    destinations = questions[question_id]["destinations"]

                    for destination in destinations:
                        ministries_list.append(destination['ministry'])

                    destinations_lengths_list.append(len(destinations))

                if field_name == 'date':
                    date = questions[question_id]["date"]
                    datetime_obj = datetime(int(date[6:]), int(date[3:5]),
                                            int(date[:2]))
                    dates_list.append(datetime_obj)

                if field_name not in df.keys():
                    df[field_name] = [questions[question_id][field_name]]
                else:
                    df[field_name].append(questions[question_id][field_name])

    if analyze_data:
        num_ministries = len(list(Counter(ministries_list)))

        df = pd.DataFrame.from_dict(df)
        df.to_csv("data/questions.csv")

        print(Counter(destinations_lengths_list))
        print("There are {} ministries".format(num_ministries))

        print(Counter(ministries_list))
        print("There are {} questions in total".format(len(questions)))

        print(min(dates_list), max(dates_list))

    datapoints_list = np.array(datapoints_list)
    labels = np.array(labels)

    if use_just_subject_feature:
        X_train, X_test, y_train, y_test = train_test_split(datapoints_list,
                                                            labels,
                                                            stratify=labels,
                                                            test_size=test_size)
    else:
        documents_list = np.array(documents_list)

        X_train_docs, X_test_docs, y_train_docs, y_test_docs = train_test_split(
            documents_list, labels, shuffle=False, test_size=test_size)

        cv = CountVectorizer()
        X_train_docs = cv.fit_transform(X_train_docs)
        X_test_docs = cv.transform(X_test_docs)

        X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(
            datapoints_list, labels, shuffle=False, test_size=test_size)

        X_train_docs = X_train_docs.toarray()
        X_test_docs = X_test_docs.toarray()

        X_train = np.concatenate((X_train_docs, X_train_data), axis=1)

        y_train = y_train_docs

        X_test = np.concatenate((X_test_docs, X_test_data), axis=1)

        y_test = y_test_docs

    baseline_models(X_train, y_train, X_test, y_test,
                   subject_feature=use_just_subject_feature)
    fasttext_model()


if __name__=="__main__":
    main()