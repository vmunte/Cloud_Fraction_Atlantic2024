# Cloud_morphologies2024
Anna Hall (ESS 569), Je-Yun Chun (ESS 569), Katherine Mifsud (ESS 569), and Vlad Munteanu(ESS 469)

The objective of this geoscience machine learning project is to attempt to correctly identify cloud types/cloud morphologies through image based classification, such as a convolutional neural network. 

The premise of this project could be completed or replicated using various model data sources like ERA5, MERRA2, CloudSat data, COSMIC data, MLS data, CMIP6 data. For this specific project, only data from geostationary weather satellite, GOES-R 16 is used for cloud identification. Data for meteorology constraints comes from MERRA-2. 

We chose to use a deep learning convolutional neural network (CNN) to identify and classify satellite images of clouds because this will allow us to automatically look for patterns, textures, and shapes in the images; sort and label them efficiently. Our CNN will process our satellite images that we input from GOES-R 16, and look for cloud formation, shape, and brightness. These features will aid in the classification process. CNNs are composed of several layers. The input layer is coomprised of the GOES-R 16 satellite images. The convolutional layer, will process the image and identify features and patterns. The output layer will provide the final morphology classification. 

These notebooks and scripts are created in python through the VS Code platform. 
Before running any scripts or notebooks, the user should install a seperate environment following these guidelines. 
In the terminal window, `conda env create "name of environment" `
To activate the environment before running any scripts/notebooks, `conda activate "name of environment"`

Relevant packages and libraries to install include : 
`pip install tensorflow numpy matplotlib opencv-python`
`import tensorflow as tf`
`from tensorflow.keras import layers, models`
`import numpy as np`
`import cv2`
`import os`
`import matplotlib.pyplot as plt`
`from tensorflow.keras.preprocessing.image import ImageDataGenerator`


High-level descriptions of each added notebook and scripts in this repository will be updated frequently, as new files are added. 
