# Installation

## Dependencies

DeepLab depends on the following libraries:

*   Numpy
*   Pillow 
*   Matplotlib
*   Tensorflow
*   OpenCV 3.X

For OpenCv use the .sh bash scrpit and will install it on your Ubuntu 18.04 LTS without problems
from your root
```bash
# From tensorflow/models/research/deeplab
cp ./install-opencv.sh.sh ~/

cd ~/

./install-opencv.sh
```


For detailed steps to install Tensorflow, follow the [Tensorflow installation
instructions](https://www.tensorflow.org/install/). A typical user can install
Tensorflow using one of the following commands:

```bash
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
```

The remaining libraries can be installed on Ubuntu 18.04 using via Pip

```bash
sudo pip install Pillow
sudo pip install matplotlib
```

# Testing the Installation

You can test if you have successfully installed the Tensorflow DeepLab by
running the following commands:

Quick test by running webcam_normal.py:

```bash
# From tensorflow/models/research/
python webcam_normal.py
```




