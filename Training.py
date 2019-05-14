# Restrict to running on only one GPU as otherwise TensorFlow hogs them all
import imp
try:
    imp.find_module('setGPU')
    import setGPU
except ImportError:
    found = False
# ////////////////////////////////////////////////

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
from sklearn.model_selection import train_test_split
from joblib import dump
from keras.callbacks import ModelCheckpoint
from Models import create_model
from Callbacks import Plot_test

pd.set_option('display.max_columns', None)  

# (Re)create folder for plot
folders_ = ['plots_MVADist_T5qqqq', 'plots_ROCs_T5qqqq', 'plots_MVADist_T1qqqq', 'plots_ROCs_T1qqqq', 'plots_MVADist_QCD', 'plots_ROCs_QCD']
for dir in folders_:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

# Lock the random seed for reproducibility
np.random.seed = 7

# Only read the necessary variables from the trackingNtuples to conserve memory during training
read = ['trk_isTrue','trk_mva','trk_pt','trk_eta','trk_lambda','trk_dxy','trk_dz','trk_dxyClosestPV','trk_dzClosestPV',
        'trk_ptErr','trk_etaErr','trk_lambdaErr','trk_dxyErr','trk_dzErr','trk_nChi2','trk_ndof','trk_nLost',
        'trk_nPixel','trk_nStrip','trk_nPixelLay','trk_nStripLay','trk_n3DLay','trk_nLostLay','trk_algo']

# Make sure the columns of the input dataframes are in the same order as they are when you
# perform the inference using the trained network in the CMSSW
input_train = read_root('QCD_flat_small.root', columns=read)[read]

# Split a piece of the training sample out for making validation plots.
# Reset indices in the dataframes afterwards to keep them neat looking.
input_train, test_QCD = train_test_split(input_train, shuffle=True, test_size=0.1)
input_train.reset_index(drop=True, inplace=True)
test_QCD.reset_index(drop=True, inplace=True)

# Additional data from different event types to produce monitoring plots
test_T1qqqq = read_root('T1qqqqLL.root', columns=read)[read]
test_T5qqqq = read_root('T5qqqqLL.root', columns=read)[read]

input_target = input_train['trk_isTrue']
test_QCD_target = test_QCD['trk_isTrue']
test_T1qqqq_target = test_T1qqqq['trk_isTrue']
test_T5qqqq_target = test_T5qqqq['trk_isTrue']

# Drop the trk_isTrue and trk_mva variables from the training input.
# They are only needed for the monitoring plots
input_train.drop(['trk_isTrue', 'trk_mva'], inplace=True, axis=1)

# Model is defined in the Models.py file
model = create_model('Track_classifier', input_train.shape)
print model.summary()

# Callback to save the model if the validation loss improves
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

# Plotting scripts imported from Callbacks.py
plot_QCD = Plot_test((test_QCD, test_QCD_target), test_QCD, 'QCD')
plot_T1qqqq = Plot_test((test_T1qqqq, test_T1qqqq_target), test_T1qqqq, 'T1qqqq')
plot_T5qqqq = Plot_test((test_T5qqqq, test_T5qqqq_target), test_T5qqqq, 'T5qqqq')

# The main function that starts the training 
model.fit(
    input_train,
    input_target,
    batch_size=32,
    validation_split=0.1,
    epochs=500,
    callbacks=[checkpoint,plot_QCD, plot_T1qqqq, plot_T5qqqq]
)
