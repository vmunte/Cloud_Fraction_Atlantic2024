# Cloud_Fraction_Atlantic_2024
Anna Hall (ESS 569), Je-Yun Chun (ESS 569), Katherine Mifsud (ESS 569), and Vlad Munteanu (ESS 469)

The objective of this geoscience machine learning project is to develop a machine learning model and pipeline to predict total cloud cover over the Atlantic Ocean using satellite data and reanalysis data. Current global climate models (GCMs), struggle to accurately predict cloud cover (CC) due to the complex dynamics and interactions at play in cloud formation, persistence, and dissipation. Furthermore, cloud cover is an important control on climate due to the influence of clouds on the total energy budget through trapping outgoing radiation and reflecting incoming radiation. Therefore, to further constrain future climate projections, it is crucial to accurately model cloud cover. We aim to utilize reanalysis estimates of cloud cover as the ground truth to train a machine learning model to accurately predict cloud cover from other, better-constrained variables.

The premise of this project could be completed or replicated using various data sources. For this specific project, our satellite data is from the geostationary weather satellite, GOES-R 16. Specifically, we use three data products from the GOES-16 Satellite Series; a cloud image and moisture product, a cloud optical depth product, and a reflected shortwave radiation product. In addition, we use ERA5 reanalysis data for a variety of meteorological variables, including winds, temperature, humidity, and heat fluxes to further the model's ability to predict cloud cover. The data is confined to a section of Atlantic Ocean (near Bermuda), where cloud formation happens on a regular basis and rough topography that breaks up clouds is absent. To avoid biases from seasonal and diurnal variation, the data is further confined to April 2020 at 15:00:00 UTC each day. All data is re-gridded to one-quarter degree lat/lon resolution.

A variety of machine learning methods are applied, including classic machine learning (CML) models and a few deep learning models. Preliminary analysis is conducted using CML to highlight imperfections and trends in the training data in preparation for deep learning models. 

These notebooks and scripts are created in python through the VS Code platform. 
Before running any scripts or notebooks, the user should import the necessary packages listed below, should the necessary packages not be available you can try a 'pip install **package name**' 

Relevant packages and libraries to install include :

```python
from goes2go import GOES
import pandas as pd
from datetime import datetime
import xarray as xr
import subprocess
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
```
There are a couple of packages that may be unfamiliar to the every day python user **goes2go** and **subprocess**.

The **goes2go** package is developed as an easy and efficient way to access GOES satellite data on the AWS server. This package is developed by **Brian Blaylock**. Instructions on installing and using the goes2go package can be found on this GitHub page:

[Brian's GitHub](https://github.com/blaylockbk/goes2go)

[AWS CLI Download Instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

The next package that may be unfamiliar is subprocess. We use the subprocess package to access the GOES data stored on the AWS server.

To set-up the AWS CLI on your own please follow the links under **GOES on AWS**:

## References and Resources

- **Brian Blaylock’s Resources**: For accessing GOES satellite data and `goes2go` details:
  - [Brian's GitHub](https://github.com/blaylockbk/goes2go)
  - [Brian's Homepage](https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/home.html)
  - [Mapping GOES16 TrueColor](https://github.com/blaylockbk/pyBKB_v3/blob/master/BB_GOES/mapping_GOES16_TrueColor.ipynb)
  - [GOES Image Viewer](https://www.star.nesdis.noaa.gov/goes/sector.php?sat=G16&sector=gm)
  - [GOES ABI Field of View](https://goes2go.readthedocs.io/en/latest/user_guide/notebooks/field-of-view_ABI.html)

- **GOES on AWS**:
  - [NOAA GOES on AWS](https://docs.opendata.aws/noaa-goes16/cics-readme.html)
  - [GOES Series Documentation](https://www.goes-r.gov/resources/docs.html)
  - [AWS CLI Troubleshooting](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-troubleshooting.html#general-formatting)
  - [NASA GitHub for Accessing GOES Data](https://github.com/awslabs/open-data-docs/tree/main/docs/noaa/noaa-goes16)
    
- **GOES Data**:
  - [Beginner's Guide to GOES Data](https://www.goes-r.gov/downloads/resources/documents/Beginners_Guide_to_GOES-R_Series_Data.pdf)
  - [Satellite Data Information from NESDIS](https://www.star.nesdis.noaa.gov/atmospheric-composition-training/satellite_data.php#abi_aws)
  - [GOES Imager Projection](https://www.star.nesdis.noaa.gov/atmospheric-composition-training/satellite_data_goes_imager_projection.php#:~:text=If%20you%20want%20to%20work,in%20the%20GOES%2DR%20PUG.)
  - [GOES Lat/Lon Conversion Code](https://www.star.nesdis.noaa.gov/atmospheric-composition-training/python_abi_lat_lon.php)
 
- **Further Reading**:
  - [Leif’s GitHub for convml-tt](https://github.com/leifdenby/convml-tt?tab=readme-ov-file)
  - [Geostationary Projection Documentation](https://proj4.org/en/9.5/operations/projections/geos.html)
  - [Basemap Documentation](https://matplotlib.org/basemap/stable/users/geography.html)
  - [Funny Basemap Thread](https://github.com/matplotlib/basemap/issues/361)


Basemap Documentation (I did not use this, however, Brian does.)
(https://matplotlib.org/basemap/stable/users/geography.html)


**High-level descriptions of each added notebook and scripts in this repository will be updated frequently, as new files are added. 

Data is too large please navigate to this google drive for the data: https://drive.google.com/drive/folders/1PMfDE_NcJCksiyA4KFzyt387RP2d4bcR?usp=sharing

How to use this repository:
1. navigate to the notebooks folder and download the GOES data following this notebook: [DownloadGOES](https://github.com/UW-MLGEO/Cloud_fraction_Atlantic2024/blob/main/notebooks/DownloadGOESData.ipynb)
2. next navigate to the DownloadERA5 notebook to download the supplementary meteorlogical field [DownloadERA5](
