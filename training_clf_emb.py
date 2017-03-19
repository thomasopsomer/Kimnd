#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from utils import load_dataset, split_train_dev_set
from flat_dataset import parallelize_dataframe
from enron_graph_corpus import EnronGraphCorpus
from enron_graph_corpus import build_flat_dataset, create_test_df
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklean.metric import confusion_matrix
import pandas as pd
from average_precision import mapk
import xgboost as xgb


def add_features_vect(df_flat, egc, test=False, d2v=False, n2v=False,
                      ac=False):
    """ """
    rcol = "candidate" if test else "recipient"
    if n2v:
        # re - se - sim
        se_vect = np.vstack([egc.n2v[str(x)] for x in df_flat.sender])
        re_vect = np.vstack([egc.n2v[str(x)] for x in df_flat[rcol]])
        if ac:
            df_flat["se_re_sim"] = (1.0 - np.sum(se_vect * re_vect, axis=1)) / np.pi
        else:
            df_flat["se_re_sim"] = np.sum(se_vect * re_vect, axis=1)
        del se_vect, re_vect
    print "one"
    # se - msg - sim
    se_vect = np.vstack([egc.sender_rep[x] for x in df_flat.sender])
    if d2v:
        msg_vect = np.vstack([egc.d2v[x] for x in df_flat.mid])
    else:
        msg_vect = np.vstack([egc.wcbow[x] for x in df_flat.mid])
    df_flat["se_msg_sim"] = np.sum(se_vect * msg_vect, axis=1)
    del se_vect
    print "two"
    # re - msg - sim
    re_vect = np.vstack([egc.recipient_rep.get(x, np.zeros(300))
                        for x in df_flat[rcol]])
    df_flat["re_msg_sim"] = np.sum(re_vect * msg_vect, axis=1)
    del re_vect
    print "three"
    # re | se - msg - sim
    rese_vect = np.vstack([egc.recipient_sender_rep.get((getattr(row, rcol), row.sender), np.zeros(300))
                           for row in df_flat.itertuples()])
    df_flat["rese_msg_sim"] = np.sum(rese_vect * msg_vect, axis=1)
    del rese_vect
    return df_flat


def person_feature(df_flat, egc, test=False):
    """ """
    rcol = "candidate" if test else "recipient"
    first_name = df_flat[rcol].map(lambda x: x.replace("..", ".").split("@")[0].split(".")[0]).values
    msg_persons = df_flat.mid.map(lambda x: egc.doc[x]["persons"]).values
    df_flat["person_int"] = [len(set([fn]).intersection(msg_p)) for fn, msg_p in zip(first_name, msg_persons)]
    return df_flat


def sender_idx(df_flat, egc, test=False):
    """ """
    df_flat["sender_idx"] = df_flat.sender.map(lambda x: egc.mail2id[x])
    return df_flat


def date_features(df_flat):
    """ """
    df_flat["year"] = df_flat.date.map(lambda x: x.year)
    df_flat["month"] = df_flat.date.map(lambda x: x.month)
    df_flat["weekday"] = df_flat.date.map(lambda x: x.weekday())
    return df_flat


# ###


if __name__ == '__main__':

    # PATHS
    path_to_data = os.path.join(os.getcwd(), 'data/')
    path_to_d2v = "emb/d2v.model"
    path_to_temp_features = "data/time_features.csv"
    path_to_lda = "data/LDA_results.csv"

    # load training data
    ds_path = path_to_data + 'training_set.csv'
    mail_path = path_to_data + 'training_info.csv'
    df_train = load_dataset(ds_path, mail_path)

    # load test data
    ds_path = path_to_data + 'test_set.csv'
    mail_path = path_to_data + 'test_info.csv'
    df_test = load_dataset(ds_path, mail_path, train=False)

    # make corpus grpah ...
    egc = EnronGraphCorpus(df_train, df_test=df_test)
    # egc = EnronGraphCorpus(df)
    egc.build_graph()
    # load node2vec vectors
    # egc.load_n2v_vectors("emb/enron_p_0.5_q_1_u.emb")
    # load doc2vec vectors
    egc.load_d2v_vectors(path_to_d2v)
    # parse email bodies
    egc.parse_bodies()
    # fit IDF
    egc.fit_idf(min_df=4)
    # build wcbow representation with tfidf weight
    # egc.make_wcbow(w="tfidf", k=-20)
    # build representation for sender and recipient
    egc.build_people_vectors(d2v=True)
    # build outgoing recipient | sender represnetation
    egc.build_recipient_sender_vectors(d2v=True)

    # load temporal features
    temp_f = pd.read_csv(path_to_temp_features) \
        .rename(columns={"user": "sender", "contact": "recipient"})

    # load lda features
    lda_f = pd.read_csv(path_to_lda).rename(columns={"mids": "mid"})
    
    # flat training dataset with sampled paires
    df_flat_train = build_flat_dataset(
        df_train, egc.DG, egc.r_email, percent_neg=1,
        dr=False, n_cores=4)

    # flat test dataset
    df_flat_test = create_test_df(df_test, egc.DG, dr=False)

    # add features for training dataset
    df_flat_train = add_features_vect(df_flat_train, egc, test=False, d2v=True)

    # time features
    df_flat_train = df_flat_train.merge(temp_f, on=["sender", "recipient"],
                                        how="left").fillna(0.)
    # lda
    df_flat_train = df_flat_train.merge(lda_f, on="mid", how="left")
    
    # person
    df_flat_train = person_feature(df_flat_train, egc)

    # date
    df_flat_train = df_flat_train.merge(df_train[["mid", "date"]], on="mid", how="left")
    df_flat_train = date_features(df_flat_train)

    # name of features columns
    features_col = ["se_msg_sim", "re_msg_sim", "rese_msg_sim"]
    features_col = features_col + ["OUT_ratio", "IN_ratio", "OUT_ratio_recent",
                                   "IN_ratio_recent"]
    features_col = features_col + ["person_int"]
    features_col = features_col + ["0", "1", "2", "3"]
    features_col = features_col + ["weekday"]

    #
    X = df_flat_train[features_col]
    Y = df_flat_train.label

    # init random forest model
    rfr = RandomForestClassifier(
        n_estimators=50, criterion='gini', max_depth=None, n_jobs=4,
        oob_score=True)
    rfr.fit(X, Y)
    print rfr.oob_score_

    # xgboost
    # rfrxgb = xgb.XGBClassifier(max_depth=6, learning_rate=1, n_estimators=160, reg_lambda=2,
                               # objective='binary:logistic')
    # rfrxgb.fit(X, Y)

    # add features for test
    df_flat_test = add_features_vect(df_flat_test, egc, d2v=True, test=True)
    df_flat_test = df_flat_test.merge(
        temp_f, left_on=["sender", "candidate"],
        right_on=["sender", "recipient"], how="left").fillna(0.)
    df_flat_test = person_feature(df_flat_test, egc, test=True)
    # df_flat_test = sender_idx(df_flat_test, egc)`
    df_flat_test = date_features(df_flat_test)
    # lda
    df_flat_test = df_flat_test.merge(lda_f, on="mid", how="left")
    # date
    df_flat_test = df_flat_test.merge(df_test[["mid", "date"]],
                                      on="mid", how="left")
    df_flat_test = date_features(df_flat_test)

    # make presdiction
    df_flat_test["pred"] = rfr.predict_proba(df_flat_test[features_col])[:, 1]

    # aggregate prediction with the top 10 recipient if over 0.5
    tmp = df_flat_test[["mid", "sender", "candidate", "pred"]] \
        .query(df_flat_test.pred > 0.5) \
        .groupby(["sender", "mid"]).apply(lambda x: x.nlargest(10, "pred"))
    tmp = tmp[["candidate", "pred"]] \
        .reset_index().drop("level_2", axis=1) \
        .groupby(["sender", "mid"])["candidate"] \
        .apply(list) \
        .reset_index()

    # merge prediction with test dataset
    df_test2 = df_test.merge(tmp[["mid", "sender", "candidate"]], on=["mid", "sender"])

    # make the prediction file and save it
    res = df_test2[["mid", "candidate"]]
    res["recipients"] = res.candidate.map(lambda x: " ".join(x))
    res = res.drop("candidate", axis=1)
    res = res.set_index("mid")
    res.to_csv("pred.csv", index_label="mid")

