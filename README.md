# eyetrackently being built with [torch](http://torch.ch/)

Much of this code was based on the FacialKeyPoint demo available at https://github.com/nicholas-leonard/dp

# Goal
Broadly, the ultimate goal is leverage neural network to build an accessible eye-tracker based on a laptop's webcamera. This process will involve two stages

# Required Torch Packages

(Assuming torch has already been sucessfull installed)
(Also, I'm running this on Ubuntu 14.04. No other os has been tested)

For each of these, run: luarocks install [nameOfPackage]

- nn
- dp
- nngraph
- findcuda
- cunnx
- cutorch
- nnx
- camera
- image
- luasockets
- mobdebug
- sys


# To Run...

## grab data

1. Create empty folder within subfolder 'grabData' labeled data
2. 'qlua run.lua'

## Network

(network cannot be run without data. Currently, datafiles too large to be uploaded to github. Please contact Patrick (psadil@gmail.com) if you would like to test this out)

1. Clone repository: `git clone https://github.com/psadil/eyetracker.git`
2. Unpack data files within fixationCNN
3. Run CNN: 'qlua fixationdetection_conv.lua'


### NOTE: 

variables at beginning of 'fixationdetection_conv.lua' can be used to adjust the architecture of model. Some of these parameters refer to various tricks utilized by the network (normalizing color channels, dropout rate, whether to visualize certain aspects of performace, etc). Messing with those parameters shouldn't break the code. This is also where the sources of the data files are defined. HOWEVER, parameters that refer to more global workings (mostly, whether to utilize cuda; but also which data file to train on/utilize) will almost certainly break the code if everything hasn't been set up properly.

# Current Status

1. Data successfully grabbed. 
2. CNN manages to learn aspects of fixations, but MSE is currently over 500.

## TO DO

1. Train network with data from the entire screen?
2. Test with more participants?
3. Impliment eye-tracking in real-time so as to enable gaze-contingent experimentation
