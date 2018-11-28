import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import keras.backend as K
K.set_session(sess)

import pandas as pd
from root_pandas import read_root, to_root

import numpy as np
import glob
import math
import sys
import re
import os
import shutil
import hickle as hkl
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from joblib import dump

from Models import create_model
import Utils

# (Re)create folder for plot
folders_ = ['plots', 'Graph', 'plots_MVADist_T5qqqqLL', 'plots_ROCs_T5qqqqLL', 'plots_MVADist_T1qqqqLL', 'plots_ROCs_T1qqqqLL', 'plots_MVADist_QCD_Flat', 'plots_ROCs_QCD_Flat']
for dir in folders_:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

# Lock the random seed for reproducibility
np.random.seed = 7

read = ['trk_isTrue','trk_mva','trk_pt','trk_eta','trk_lambda','trk_dxy','trk_dz','trk_dxyClosestPV','trk_dzClosestPV',
        'trk_ptErr','trk_etaErr','trk_lambdaErr','trk_dxyErr','trk_dzErr','trk_nChi2','trk_ndof','trk_nLost',
        'trk_nPixel','trk_nStrip','trk_nPixelLay','trk_nStripLay','trk_n3DLay','trk_nLostLay','trk_algo']

data_train = read_root('QCD_flat_small.root', columns=read)
test1 = read_root('T1qqqqLL.root', columns=read)
test2 = read_root('T5qqqqLL.root', columns=read)

data_target = data_train['trk_isTrue']
test1_target = test1['trk_isTrue']
test2_target = test2['trk_isTrue']

data_train.rename(columns={'trk_dzClosestPV': 'trk_dzClosestPVClamped'}, inplace=True)
data_train.loc[:, 'trk_dzClosestPVClamped'] = np.clip(data_train.loc[:, 'trk_dzClosestPVClamped'], a_min=-2.0, a_max=2.0)
data_train.drop(['__array_index', 'index', 'trk_isTrue'], inplace=True, axis=1)

test1.rename(columns={'trk_dzClosestPV': 'trk_dzClosestPVClamped'}, inplace=True)
test1.loc[:, 'trk_dzClosestPVClamped'] = np.clip(data_train.loc[:, 'trk_dzClosestPVClamped'], a_min=-2.0, a_max=2.0)
test1.drop(['__array_index', 'index', 'trk_isTrue'], inplace=True, axis=1)

test2.rename(columns={'trk_dzClosestPV': 'trk_dzClosestPVClamped'}, inplace=True)
test2.loc[:, 'trk_dzClosestPVClamped'] = np.clip(data_train.loc[:, 'trk_dzClosestPVClamped'], a_min=-2.0, a_max=2.0)
test2.drop(['__array_index', 'index', 'trk_isTrue'], inplace=True, axis=1)


scaler = MinMaxScaler().fit(data_train.values)
dump(scaler, 'scaler.pkl')

data_train = pd.DataFrame(scaler.transform(data_train), columns=data_train.columns)
test1 = pd.DataFrame(scaler.transform(test1), columns=test1.columns)
test2 = pd.DataFrame(scaler.transform(test2), columns=test2.columns)

model = create_model('Track_Classifier', data_train.shape)

print model.summary()

model.fit(
    data_train,
    data_target,
    batch_size=32,
    validation_split=0.1,
    epochs=500
)
