#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 15:23:38 2018

@author: elvex
"""

from hmmlearn.hmm import GMMHMM as HMM
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from os.path import basename, abspath
from sklearn.mixture import GaussianMixture


def __wav2mfcc(path):
    (rate,sig) = wav.read(path)
    mfcc_feat = mfcc(sig,rate)
    return mfcc_feat


def dir2mfcc(path):
    path = abspath(path)
    lst = glob(path + "/*.wav")
    d_test, d_train = {}, {}
    for f in lst:
        n = basename(f).split(".")[0]
        if "test" in n: 
            d_test[n] = __wav2mfcc(f)
        elif "train" in n:
            d_train[n] = __wav2mfcc(f)
        else:
            pass
    return d_train, d_test


def train_dic_monoMachine(d_train, nb_gauss = 2):
    d_lbl = {e.split("_")[0] : i for i,e in enumerate(d_train)}
    X, Y = [], []
    for f in d_train:
        n = d_train[f].shape[0]
        X.append(d_train[f])
        Y.append(np.ones((n, 1)) * d_lbl[f.split("_")[0]])
    X, Y = np.concatenate(X), np.concatenate(Y)
    d_lbl = {i : e.split("_")[0] for i,e in enumerate(d_train)}
    M = GaussianMixture(nb_gauss)
    M.fit(X, Y)
    return (M, d_lbl)


def test_dic(M, d_lbl, d_test):
    for t in d_test:
        l = list(M.predict(d_test[t]))
        occ, val_freq = max(map(lambda val: (l.count(val), val), set(l)))
        classe_trouvee =  d_lbl[val_freq]
        veracite = "raison" if classe_trouvee == t.split("_")[0] else "tort"
        taux = (occ / len(l)) * 100
        print("La classe de '{}' semble être '{}', à {}. {} % des résultats confirment cette conjecture".format(
                t, classe_trouvee, veracite, round(taux, 2)))
    
        
    
        
    