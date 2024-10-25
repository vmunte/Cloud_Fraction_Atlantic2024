# Cloud_morphologies2024
Anna Hall (ESS 569), Je-Yun Chun (ESS 569), Katherine Mifsud (ESS 569), and Vlad Munteanu(ESS 469)

The objective of this geoscience machine learning project is to attempt to correctly identify cloud types/cloud morphologies through image based classification, such as a convolutional neural network. 

The premise of this project could be completed or replicated using various data sources. For this specific project, only data from geostationary weather satellite, GOES-R 16 is used for cloud identification. Data for meteorology constraints comes from MERRA-2. 

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
`from goes2go import GOES`
`import pandas as pd`
`from datetime import datetime`
`import xarray as xr`
`import subprocess`
`from netCDF4 import Dataset`
`import cartopy.crs as ccrs`
`import cartopy.feature as cfeature`

The following are instructional links on how we downloaded data, organized it, and began processing it with the help of Brian Baylock, whom we acknowledge for helping us download the data. 

need to set up the AWS CLI

Satellite Data Information from NESDIS
(https://www.star.nesdis.noaa.gov/atmospheric-composition-training/satellite_data.php#abi_aws)

base aws registry
(https://registry.opendata.aws/noaa-goes/)

nasa github for accessing GOES data
(https://github.com/awslabs/open-data-docs/tree/main/docs/noaa/noaa-goes16)

NESDIS STAR code for converting between ABI Grid and Lat/Lon
(https://www.star.nesdis.noaa.gov/atmospheric-composition-training/python_abi_lat_lon.php)

GOES Image Viewer
(https://www.star.nesdis.noaa.gov/goes/sector.php?sat=G16&sector=gm)

Leif's Github
(https://github.com/leifdenby/convml-tt?tab=readme-ov-file)

beginner's guide to GOES data
(https://www.goes-r.gov/downloads/resources/documents/Beginners_Guide_to_GOES-R_Series_Data.pdf)

NOAA GOES on AWS
(https://docs.opendata.aws/noaa-goes16/cics-readme.html)

GOES documents -- PUG v5 for converting to lat/lon grid
(https://www.goes-r.gov/resources/docs.html)

Imager Projector 
(https://www.star.nesdis.noaa.gov/atmospheric-composition-training/satellite_data_goes_imager_projection.php#:~:text=If%20you%20want%20to%20work,in%20the%20GOES%2DR%20PUG.)

Brian's Homepage
(https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/home.html)

Brian's Github
(https://github.com/blaylockbk/pyBKB_v3/blob/master/BB_GOES/mapping_GOES16_TrueColor.ipynb)

Brian's Bulk download github (GOES2go)
(https://github.com/blaylockbk/goes2go?tab=readme-ov-file)

Brian's ABI field of view
(https://goes2go.readthedocs.io/en/latest/user_guide/notebooks/field-of-view_ABI.html)

geostationary projection documentation
(https://proj4.org/en/9.5/operations/projections/geos.html)

funny Basemap Thread
(https://github.com/matplotlib/basemap/issues/361)

Basemap Documentation (I did not use this, however, Brian does.)
(https://matplotlib.org/basemap/stable/users/geography.html)

for debugging AWS CLI follow this link:
(https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-troubleshooting.html#general-formatting)


**High-level descriptions of each added notebook and scripts in this repository will be updated frequently, as new files are added. 
