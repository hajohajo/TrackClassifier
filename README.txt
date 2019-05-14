An example repository to show how to train a DNN for track classification.

The training is done using Keras python library, using TensorFlow as the backend for the heavy lifting. The required python packages are stored in "requirements.txt", and in principle command 

"pip install -r requirements.txt"

should take care of installing the needed packages. This does require however a working installation of pyroot (root-pandas package relies on it) for reading and writing ROOT files. Use of virtual environments in order not to mess up whatever python packages there already are is strictly recommended.

Training.py
------------
Contains the main code, where the data is read, input variables are chosen and arranged into pandas dataFrames that are then fed to the training loop.
Two test sets are defined from different type of events, that can be used to monitor how the training progresses. Note that the training sample included
in this repository is too small to do any deployable training for a model, as the networks require millions of examples to result in high quality models.

Executing this file with 

'python Training.py'

runs the main program, producing monitoring plots and saving checkpoints of the trained model every tenth epoch.

Utils.py
--------
Contains a helper enumerator for keeping track of the track iterations.

Callbacks.py
------------
A quick and dirty callback for plotting the monitoring plots during training. One can define their own callbacks
following the example in this file.


*****
After the training
*****

After the training has converged (the epochs ran out or training is otherwise stopped), one has to still convert the produced 'model.h5' file into a format
that the CMSSW can understand. Current implementation of track classifiers supports the LWTNN[1] framework for performing the inference. For that the model
saved during training has to be split into architechture and weight files by executing 'splitModelToArchAndWeights.py'. (This could be done cleaner by just
making a custom callback for the training that saves the model directly into these separate files instead of using the Keras default ModelCheckpoint).

With these files one can now follow the guide in LWTNN documentation to create a .json file that can be then used inside the CMSSW. Briefly summarized one
has to run the kerasfunc2json.py script to create the .json and modify the input variable names to match the names used in the part of the CMSSW performing
the inferences.

An example of what the output .json should look like can be found at [2]. This may be useful if there is some issues in the formatting or how to name the variables.

On the CMSSW side the important code is at the RecoTracker/FinalTrackSelector subpackage. From there:
[3.1] defines which .json contains the desired DNN model.
[3.2] contains the ESProducer for the lwtnn network.
[3.3] contains most of the code of interest, here the input variables (and their order!) to the network are defined
[3.41, 3.42] has the TrackMVAClassifierBase class that the trackLWTNNClassifier builds on 

There is a process modifier [4] that can be used in the configuration file to turn on use of DNN classifier for all steps of the IterativeTracking in phase1,
or one can modify the desired step configuration files in RecoTracker/IterativeTracking to turn on DNN manually for only some of the steps.



Links
----------------------------------
[1] https://github.com/lwtnn/lwtnn
[2] https://github.com/cms-data/RecoTracker-FinalTrackSelectors
[3.1] https://github.com/cms-sw/cmssw/blob/master/RecoTracker/FinalTrackSelectors/python/trackSelectionLwtnn_cfi.py
[3.2] https://github.com/cms-sw/cmssw/blob/master/RecoTracker/FinalTrackSelectors/plugins/LwtnnESProducer.cc
[3.3] https://github.com/cms-sw/cmssw/blob/master/RecoTracker/FinalTrackSelectors/plugins/TrackLwtnnClassifier.cc
[3.41] https://github.com/cms-sw/cmssw/blob/master/RecoTracker/FinalTrackSelectors/src/TrackMVAClassifierBase.cc
[3.42] https://github.com/cms-sw/cmssw/blob/master/RecoTracker/FinalTrackSelectors/interface/TrackMVAClassifier.h
[4.0] https://github.com/cms-sw/cmssw/blob/master/Configuration/ProcessModifiers/python/trackdnn_cff.py
