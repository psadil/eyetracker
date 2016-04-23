# eyetrackently being built with [torch](http://torch.ch/)

Code sourced primarily from examples provided by [e-lab's demos](https://github.com/e-lab/torch7-demos). See (online lecture)[https://www.youtube.com/watch?v=BCoGFXPPYxk&index=18&list=PLNgy4gid0G9e0-SiJWEdNEcHUYjo1T5XL] for a demonstration.

Additionally, data for training comes from [SynthesEyes](http://www.cl.cam.ac.uk/research/rainbow/projects/syntheseyes/)

# Goal
Broadly, the ultimate goal is leverage neural network to build an accessible eye-tracker based on a laptop's webcamera. This process will involve two stages

1. Train a convolutional neural network capable of reliably discriminate between photos that contain eyes and not eyes.
..*Call this first network the 'eye-detector.'
2. Train a second convolutional network on the dataset provided by [SynthesEyes](http://www.cl.cam.ac.uk/research/rainbow/projects/syntheseyes/), to estimate fixation point from eye position.
..* This network will eventually receive as input the first network (stripped of the classification layer)
..* Call this second network the 'fixation-detector'
3. Use this trained network in real-time to pick out the eyes from the feed of the camera
..* Call this the 'tracker'
..*This will be similar to the face-detector demonstrated by e-labs. Except (clearly), the underlying model is trained to detect eyes, not faces.
4. Input the pixels targeted by the eye-detector as containing an eye to the fixation-detector, and estimate fixation location.

# Required Torch Packages

(Assuming torch has already been sucessfull installed)
(Also, I'm running this on Ubuntu 14.04. No other os has been tested)

For each of these, run: luarocks install [nameOfPackage]

- nn
- nngraph
- findcuda
- cunnx
- nnx
- camera
- image
- luasockets
- mobdebug
- sys


# To Run

## Set-up
1. Clone repository: `git clone https://github.com/psadil/eyetracker.git`
2. Unpack data files
..* detailed explanation of required path structure to come. Not currently writing because it is expected to change rapidly. Also, if all goes well, the massive data files won't need to be uploaded to git anyway...
..* Also, you'll need to go checkout [SynthesEyes](http://www.cl.cam.ac.uk/research/rainbow/projects/syntheseyes/) to get the eye dataset

### CODE MGHT NOT RUN WITHOUT PROPER POSITIONING OF DATAFILES

## Train eye-detector

1. Navigate to train-eye-detector subfolder within your new, local repository.
2. Train eye detector by running in terminal: `qlua run.lua`


## Train fixation-detector

To be written as the code is written... 

## Run tracker

1. Navigate to eye-detector subfolder within this repository.
2. Execute `qlua run.lua`

## NOTES

The general pattern is that the different stages of these models are called by executing: `qlua run.lua` within one of the subfolders. Each of these 'run.lua' files begin with a set of optional parameters. Some of these parameters refer to various tricks utilized by the network (normalizing color channels, dropout rate, whether to visualize certain aspects of performace, etc). Messing with those parameters shouldn't break the code. This is also where the sources of the data files are defined. HOWEVER, parameters that refer to more global workings (mostly, whether to utilize cuda; but also which data file to train on/utilize) will almost certainly break the code if everything hasn't been set up properly.


# Current Status

## eye-detector

Sucessfully trained on the SynthesEyes data! Model gets about 99% accuracy on both the background and eye image classifications. However, based on it's performance within the Tracker, the model has difficulty generalizing to new eyes/background. 

NOTE: face-detector model performs beautifully

### Currently trying...

1. Implementing with deeper network
..*As of writing, there was only 1 convoluational network before classification
2. Tweaking scale parameters in Tracker

## fixation-detector

Need to build...

## Tracker

Seems to work! The scale parameters for the pyramiding might be a bit off (meaning the model isn't searching the camera frame with a usefully sized kernal), so that might need some work. But, there are positions in which the Tracker is only picking out my eye from the image.

### TO DO

1. figure out how to display do indicating model's prediction of fixation