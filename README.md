# fake-face-detector


### Project Overview
- Load and preprocess the dataset of fake and real images
- Train the image classifier on the dataset of fake and real images
- Use the trained classifier to predict if an image of a face was created by AI, or a photo taken of a real person

### Data
- The data used specifically for this project is not organized into any one dataset online.
- The dataset used contains some images from popular online datasets such as {The 1 million Fake Faces, UTKFace, Flickr-Faces-HQ Dataset} and some random images from google.

- The data was into comprised 3 folders: test, train, validate

- Inside the train, test and validate folders there should be folders bearing a specific number which corresponds to a specific category
- For example, if we have the image x.jpg and it is a lotus it could be in a path like this /test/real/x.jpg

### Install(run on command line)
- pip install pandas
- pip install matplotlib
- pip install Pillow
- pip install requests
- In order to install Pytorch follow the instructions given on the [official site](https://pytorch.org/)

### Running the command line application
- Train a new network on a data set with **train.py**
  - Basic Usage : ```python train.py data_directory```<br/>
  - Prints out current epoch, training loss, validation loss, and validation accuracy as the netowrk trains
  - Options:
    - Set direcotry to save checkpoints: ```python train.py data_dir --save_dir save_directory```
    - Set hyperparameters: ```python train.py data_dir --learning_rate 0.001 --epochs 2```
    - Use GPU for training: ```python train.py data_dir --gpu gpu```
  - Output: A trained network ready with checkpoint saved for doing parsing of face images and identifying the "realness" of the face.
    
- Predict an image class with **predict.py** along with the probability. That is you'll pass in a single image /path/to/image
  - Basic usage: ```python predict.py /path/to/image checkpoint```
  - Options:
    - Return most likely class: ```python predict.py input checkpoint```
    - Use a mapping of categories to real names: ```python predict.py input checkpoint```
    - Use GPU for inference: ```python predict.py input checkpoint --gpu```
  - Output: The probability a face is real or fake
  
- Predict an image class using openVino inference with **openVino_predict.py** along with the probability. That is you'll pass in one or more images /path/to/image
  - Basic usage: ```python openVino_predict.py -m /path/to/model.onnx -i /path/to/image1 [/path/to/image2 ..] --labels /path/to/labels```
  - Options:
     - -h, --help       Show this help message and exit.
     - -m MODEL, --model MODEL
                        Required. Path to an .xml or .onnx file with a trained model.
     - -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Required. Path to a folder with images or path to an
                        image files
     - -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional. Required for CPU custom layers. MKLDNN (CPU)-targeted custom layers.
                        Absolute path to a shared library with the kernels
                        implementations.
     - -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The sample
                        will look for a suitable plugin for device specified.
                        Default value is CPU
     - -labels LABELS       Optional. Path to a labels mapping file
     - -nt NUMBER_TOP, --number_top NUMBER_TOP
                        Optional. Number of top results
  - Output: The probability a face is real and fake
