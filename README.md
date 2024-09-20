# Note
The lib is still under development and is not fully usable at this time

# General Info
Library to process, format, and analyze BlueDrop accelerometer data. The main functions and classes are in the lib folder. An example is provided using the library for the analysis of a folder of BlueDrop data.

[![Documentation Status](https://readthedocs.org/projects/bluedrop-analysis-lib/badge/?version=latest)](https://bluedrop-analysis-lib.readthedocs.io/en/latest/?badge=latest)
[![Python application (Conda)](https://github.com/UF-Coastal-Geotech/BlueDrop_Analysis_Lib/actions/workflows/test-conda.yml/badge.svg)](https://github.com/UF-Coastal-Geotech/BlueDrop_Analysis_Lib/actions/workflows/test-conda.yml)

# Instructions to set up BlueDrop Analysis on your computer

### Purpose
The purpose of these instructions are to set up BlueDrop Analysis on your own computer. These instructions are subject to change in the future. The goal for this repo is to change it to be a python package hosted through ```conda```.

### Necessary downloads

The necessary downloads to use this code are:

1) An IDE that can run jupyter notebooks (VS Code is what I recommend, VS Code Download [link](https://code.visualstudio.com/download))
2) git, [link](https://git-scm.com/downloads)
3) Anaconda (or miniconda), miniconda [link](https://docs.anaconda.com/miniconda/miniconda-install/)


### Steps to get started with BlueDrop Analysis Lib

1) Clone the repo to a local folder using the following command in your terminal: ```git clone https://github.com/UF-Coastal-Geotech/BlueDrop_Analysis_Lib.git```

2) Use VS Code to open the newly cloned ```BlueDrop_Analysis_Lib``` folder

3) Using the side menu view the ```Examples``` folder and open the ```Example_PFFP_walkthrough.ipynb``` file. This is a jupyter notebook and walks through some of the common steps and functions that are used to process PFFP data files.

4) On ```Windows``` open the ```Anaconda Prompt``` or on ```Linux``` open a terminal of your choice that has conda added to it.

#### Note
The purpose of the next couple of steps are to use ```conda``` and the ```environment.yml``` file to install all the necessary packages and modules to run the all the functions in the library.

5) Using the terminal change your working directory to the ```BlueDrop_Analysis_Lib``` folder. 

6a) If on windows run the following command in the ```conda``` terminal: 
```conda env create --file environment_win.yml --name PFFP_Analysis```

6b) If on linux run the following command:
```conda env create --file environment.yml```

Installing the packages may take a few minutes.

8)  Activate the ```PFFP_Analysis``` ```conda``` environment. 
    * The easiest way to do this in VS Code is to try running a cell in ```Example_PFFP_walkthrough.ipynb```. VS Code should ask you which kernel you want to use. Choose ```Python Environments``` > ```PFFP_Analysis```

9)  Once the Python Kernel is selected and the cells in ```Example_PFFP_walkthrough.ipynb``` are running you are good to go.

