#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, MinMaxScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, 'data', "german_credit_data_biased_training.csv")
MODEL_PATH = os.path.join(BASE_DIR, 'credit_risk', 'model.pkl')


def scoring(payload_scoring, model, X_prs, y_prs):
    logging.info('starting scoring')
    js = json.loads(json.dumps(payload_scoring.get('input_data', '')))
    if isinstance(js, str):
        js = js.replace('\'', '\",')
        js = json.loads(js)
    fields = js.get('fields', []) + ["PREDICTION", "PROBABILITY"]
    X_test = np.array(js.get('values', []))
    # X_test_processed = np.empty((X_test.shape[0], X_test.shape[1] - 1))
    X_test_processed = []

    X_test_right_format=[[] for i in range(X_test.shape[0])]
    for i in range(X_test.shape[1]):
        if isinstance(X_prs[i], LabelEncoder):
            x = X_prs[i].transform(X_test[:, i].astype(str).reshape((-1, 1)))
            X_test_processed.append(list(x.reshape((-1,))))
            [X_test_right_format[j].append(X_test[j,i]) for j in range(len(x))]
        else:
            x = X_prs[i].transform(X_test[:, i].reshape((-1, 1)))
            X_test_processed.append(list(x.reshape((-1,))))
            [X_test_right_format[j].append(int(X_test[j,i])) for j in range(len(x))]
    X_test_processed = np.array(X_test_processed).T
    y_prob = model.predict_proba(X_test_processed)
    y_pred = model.predict(X_test_processed)

    values = [X_test_right_format[i] +
              list(y_prs.inverse_transform(y_pred[i].reshape((-1)))) +
              [list(y_prob[i, :])] for i in range(len(X_test))]
    return {"fields": fields,
            "labels": list(y_prs.classes_.reshape((-1))),
            "values": values}


def training(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model


def preprocess(df):
    preprocess_config = {'LOANDURATION': KBinsDiscretizer(n_bins=5, encode='ordinal'), 'LOANAMOUNT': MinMaxScaler(),
                         'AGE': KBinsDiscretizer(n_bins=5, encode='ordinal'), 'INSTALLMENTPERCENT': MinMaxScaler(),
                         'CURRENTRESIDENCEDURATION': MinMaxScaler(), 'EXISTINGCREDITSCOUNT': MinMaxScaler(),
                         'DEPENDENTS': MinMaxScaler()}
    preprocessors = []
    columns = df.columns.tolist()
    X_train = []

    for i in range(len(columns)):
        if columns[i] not in preprocess_config.keys():
            p = LabelEncoder()
            y = p.fit_transform(df.values[:, i].astype(str).reshape((-1, 1)))
        else:
            p = preprocess_config[columns[i]]
            y = p.fit_transform(df.values[:, i].reshape((-1, 1)))
        X_train.append(list(y.reshape((-1,))))
        preprocessors.append(p)
    return preprocessors, np.array(X_train).T


def get_scoring_payload(df, no_of_records_to_score=1):
    fields = df.columns.tolist()
    values = df[fields].values.tolist()

    payload_scoring = {"fields": fields, "values": values[:no_of_records_to_score]}
    return payload_scoring


if __name__ == '__main__':
    label_column = "RISK"
    model_type = "binary"

    data_df = pd.read_csv(file_path)
    y_prs = LabelEncoder()
    y_train = y_prs.fit_transform(data_df[label_column])

    cols_to_remove = [label_column]
    for col in cols_to_remove:
        if col in data_df.columns:
            del data_df[col]

    payload_scoring = get_scoring_payload(data_df, 2)
    print(payload_scoring)

    X_prs, X_train = preprocess(data_df)

    model = training(X_train, y_train)
    with open(MODEL_PATH, 'wb') as f:
        r = (model, X_prs, y_prs)
        pickle.dump(r, f)

    sample_test_payload = {
        'fields': ['CHECKINGSTATUS', 'LOANDURATION', 'CREDITHISTORY', 'LOANPURPOSE', 'LOANAMOUNT',
                   'EXISTINGSAVINGS', 'EMPLOYMENTDURATION', 'INSTALLMENTPERCENT', 'SEX', 'OTHERSONLOAN',
                   'CURRENTRESIDENCEDURATION', 'OWNSPROPERTY', 'AGE', 'INSTALLMENTPLANS', 'HOUSING',
                   'EXISTINGCREDITSCOUNT', 'JOB', 'DEPENDENTS', 'TELEPHONE', 'FOREIGNWORKER'], 'values': [
            ['0_to_200', 31, 'credits_paid_to_date', 'other', 1889, '100_to_500', 'less_1', 3, 'female', 'none', 3,
             'savings_insurance', 32, 'none', 'own', 1, 'skilled', 1, 'none', 'yes'],
            ['less_0', 18, 'credits_paid_to_date', 'car_new', 462, 'less_100', '1_to_4', 2, 'female', 'none', 2,
             'savings_insurance', 37, 'stores', 'own', 2, 'skilled', 1, 'none', 'yes']]}

    with open(MODEL_PATH, 'rb') as f:
        res = pickle.load(f)

    print(scoring({'input_data': sample_test_payload}, *res))
