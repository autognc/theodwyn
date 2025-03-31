# theodwyn
This repository provides the software stack for the Eomer and Eowyn Unmanned Ground Vehicles (UGVs) at the Controls for Distributed and Uncertain Systems Lab (CDUS). The Eomer stack, unlike Eowyn, is equipped with a vision-sensor and performs relative navigation with respect to a given target -- usually, Eowyn. Details on the UGVs are provided below for reference.

The bill of materials and reference images for both Eomer and Eowyn UGVs are found [here]()

### **Table of Contents**    
  * [1 | Requirements](#1--requirements)
  * [2 | Installation](#2--installation)
  * [3 | Functionality](#3--functionality)
  * [4 | Usage](#4--usage)
  * [5 | Validation](#5--validation)

## 1 | Requirements 
The package in this repository has been developed and tested in **Python 3.8** and requires **python>3.8** due to associated functionality introduced in version 3.8. This repository additionally depends on the following packages, some of which we recommend be built from source:

###  1.1 | Basic Functionality 
1. [rohan](https://github.com/PeteLealiieeJ/rohan)
```console
pip install git+https://github.com/PeteLealiieeJ/rohan.git
```

2. [numpy](https://pypi.org/project/numpy/)
```console
$ python -m pip install numpy
```

### 1.2 | Computer Vision Package
3. [opencv-python](https://pypi.org/project/opencv-python/)

```console
$ python -m pip install opencv-python
```
> [!WARNING] 
> This repository includes implementation for video streaming with Gstreamer inside of OpenCV and that functionality requires that OpenCV be [built from source](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md) with the cmake flag `-D WITH_GSTREAMER=ON`


### 1.3 | Network Protocol Package
4. [pyzmq](https://pyzmq.readthedocs.io/en/latest/)
```console
$ python -m pip install pyzmq
```
> [!WARNING]
> If users intend to communicate via a UDP connection i.e. use the ZMQDish and ZMQRadio modules and child modules provided, they must install [zmq from sourcewith draft socket support](https://pyzmq.readthedocs.io/en/latest/howto/draft.html).


### 1.4 | Controller Package
5. [pygame](https://pypi.org/project/pygame/)
```console
$ python -m pip install pygame
```

### 1.5 | PWM Expansion I2C Package
6. [adafruit-circuitpython-servokit](https://pypi.org/project/adafruit-circuitpython-servokit/)
```console
$ python -m pip install adafruit-circuitpython-servokit
```


### 1.6 | Camera API Packages
7. [pyrealsense2](https://pypi.org/project/pyrealsense2/)
```console
$ python -m pip install pyrealsense2
```
> [!WARNING] 
> We have found that the intel realsense libraries and asscoiated python bindings usually need to be [built from source](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md), especially on microcomputers

8. [ximea](https://www.ximea.com/support/wiki/apis/ximea_linux_software_package)
```console
cd <INSTALL DIR>
wget https://kb.ximea.com/downloads/recent/XIMEA_Linux_ARM_SP.tgz
tar xzf XIMEA_Linux_ARM_SP.tgz
cd package
./install
```
> [!NOTE] 
> The instruction proved above assumes you are installing for a machine operating with a Linux OS and ARM architecture. More info and instruction for other systems can be found on [ximea's api support website](https://www.ximea.com/support/wiki/apis/ximea_linux_software_package)

## 2 | Installation
To install the python package, the following is additionally required:

9. [setuptools](https://pypi.org/project/setuptools/)

```console
$ python -m pip install --upgrade setuptools --upgrade packaging --upgrade wheel
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

###  2.2 | PyPI installation 
- [ ] NOT YET PUBLISHED ON PYPI

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
