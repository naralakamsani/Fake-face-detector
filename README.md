# fake-face-detector


### Project Overview
- Load and preprocess the dataset of fake and real images
- Train the image classifier on the dataset of fake and real images
- Use the trained classifier to predict if an image of a face was created by AI, or a photo taken of a real person

### Data
- The data used specifically for this project is not organized into any one dataset online.
- The dataset used contains some images from popular online datasets such as {The 1 million Fake Faces, UTKFace, LFW} and some random images from google.

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
  
- Predict an image class with **openVino_predict.py** along with the probability. That is you'll pass in a single image /path/to/image
