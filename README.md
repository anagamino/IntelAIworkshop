# UConn Intel/IBACS AI Workshop
This is the repo that will be used to store the code used 
for the Intel / IBACs AI technical workshop hosted at 
the University of Connecticut.

In order to run the scripts in this directory, you will 
need python 3.6 and several packages.
Required packages:
+ TensorFlow 1.7
+ PIL (via Pillow)
+ OpenCV

Through downloading and installing TensorFlow you will download 
the other packages not listed.

In order to install TensorFlow, please refer to the official installation
instructions located [here](https://www.tensorflow.org/install/).
I highly recommend following the installation instructions that create
a virtual environment in the *~/tensorflow* folder. If you are on a laptop,
you will most likely want to install the *cpu* version of TensorFlow.

In order to install PIL, first activate the TensorFlow virtual environment.
You can do so by moving to the *~/tensorflow* directory via the terminal then
typing *source ./bin/activate*.
Once you have activated your TensorFlow environment,
run *pip install Pillow* in your terminal.

OpenCV may be installed by running *pip install opencv-python* in your
TensorFlow environment.

The current version of this directory contains four primary scripts:
+ keras_cifar10.py
+ keras_cifar10.ipynb
+ tf_cifar10.py
+ grad_cam.py

The keras_cifar10.* scripts build and train a multi-layer convolutional
neural network designed to classify the 10-category version of the cifar dataset.
These scripts make use of the high-level API [**Keras**](https://keras.io),
which is included in and serves as a front-end to the TensorFlow API. 

The first two scripts scripts are duplicates of one another except that
the code is chunked in the .ipynb file to facilitate understanding of what
each section of code is doing. 

The tf_cifar10.py script builds the same network constructed
in the previous scripts, but using a slightly lower-level API. 

The final script, grad_cam.py, creates class activation maps
for a given input picture. These help you visualize what about the input
the network uses to make its classification. The gradient class activation
map provides a course measure of activation, while the guided gradient
class activation map provied a more fine grain measure of activation.

The scripts located in this directory have been tested with Python 3.6
and TensorFlow 1.7.