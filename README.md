# theodwyn
This repository provides the software stack for the Eomer and Eowyn Unmanned Ground Vehicles (UGVs) at the Conrtols for Distributed and Uncertain Systems Lab (CDUS). The Eomer stack, unlike Eowyn, is equipped with a vision-sensor and performs relative navigation with respect to a given target -- usually, Eowyn. The UGVs are pictured below for reference:

- [ ] TODO: Image of Eomer
- [ ] TODO: Image of Eowyn

 
The bill of materials for both Eomer and Eowyn UGVs are found [here]()

### **Table of Contents**    
  * [1 | Requirements](#1--requirements)
  * [2 | Installation](#2--installation)
  * [3 | Functionality](#3--functionality)
  * [4 | Usage](#4--usage)
  * [5 | Validation](#5--validation)

## 1 | Requirements 
###  1.1 | Basic Functionality 
The basic functionality in this repository requires **Python 3.7+** and depends on the following packages:

1. [rohan]()

2. [numpy](https://pypi.org/project/numpy/)

```console
$ python -m pip install numpy
```

2. [opencv-python](https://pypi.org/project/opencv-python/)

> [!WARNING]
> If users intend to stream video data using opencv they must instead build opencv [from source](https://github.com/opencv/opencv) and enable gstreamer support in the cmake configuration; However, users can attempt to install the necessary packages via pip in the following

```console
$ python -m pip install opencv-python
```
> [!WARNING] 
> We implemented video streaming with Gstreamer inside of OpenCV and that functionality requires that OpenCV be [built from source](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md), indicating the cmake flag `-D WITH_GSTREAMER=ON`

###  1.2 | Debug and Validation Models 
To replicate validations and use the provided models for control, communication and video streaming, the following packages are required in addition:

3. [pygame](https://pypi.org/project/pygame/) <-- for interfacing gamepads
4. [adafruit-circuitpython-servokit](https://pypi.org/project/adafruit-circuitpython-servokit/) <-- for sending pwm signals via adafruit IIC

```console
$ python -m pip install pygame adafruit-circuitpython-servokit
```

5. [pyrealsense2](https://pypi.org/project/pyrealsense2/) <-- for streaming intel realsense camera channels 
> [!WARNING] 
> We have found that the intel realsense libraries and asscoiated python bindings usually need to be [built from source](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md), especially on microcomputers; However, users can attempt to install the necessary packages via pip in the following

```console
$ python -m pip install pyrealsense2
```

## 2 | Installation
To install the python package, the following is additionally required:

6. [setuptools](https://pypi.org/project/setuptools/)

```console
$ python -m pip install setuptools
```

###  2.1 | Local Installation 
As this project is not yet uploaded to the Python Package Index (PyPI) it is, many times, useful to install this package locally via pip:

```ShellSession
$ cd <WORKSPACE>
$ git clone <GITREPO> -b <VERSION>
$ cd theodwyn
$ python -m pip install .
```

where `<WORKSPACE>` location is up to the user and a `<VERSION>` can be found in the the tags list on the associated github page

###  2.2 | Pip installation 
- [ ] NOT YET IMPLEMENTED

## 3 | Functionality
- [ ] TODO

## 4 | Usage 
- [ ] TODO

## 5 | Validation
As this package handles interactions between hardware components, our validation procedure is carried out through a baseline system in house. In many ways, this makes it difficult to validate contributions. However, when possible, we validate code on the pan-tilt camera system -- pictured below -- using pytest and the following debug files found in this repository:

1. `theodwyn/config/debug_config.json`
2. `tests/`

Validation of changes to the base classes is then done through pytest with the following:

```console
$ cd <WORKSPACE>/theodwyn
$ pytest
```