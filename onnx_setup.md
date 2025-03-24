# ONNX Runtime and CUDA Installation on Jetson Orin Nano

This documentation provides step-by-step instructions for correctly installing ONNX Runtime with GPU support and CUDA on a Jetson Orin Nano **-Hamza Mujtaba**

## Step 1: Remove Existing CUDA and cuDNN Packages

First, completely remove any previously installed CUDA and cuDNN packages:

```bash
sudo apt-get --purge remove "*cublas*" "cuda*"
sudo apt-get autoremove
sudo apt-get autoclean
sudo apt-get --purge remove "*cudnn*"
sudo apt-get autoremove
sudo apt-get autoclean
```

Verify removal:

```bash
dpkg -l | grep nvidia
```

There should be no cuda or cudnn packages listed.

## Step 2: Install CUDA 12.0 and cuDNN

Download and add CUDA keyring:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
```

Install CUDA 12.0 and cuDNN libraries:

```bash
sudo apt-get install cuda-12-0
sudo apt-get install libcudnn8 libcudnn8-dev
```

### Configure Environment Variables

Edit your `~/.bashrc` file to include the following (if not already present):

```bash
export PATH="/usr/local/cuda-12.0/bin:/home/<user>/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH"
```

Apply changes:

```bash
source ~/.bashrc
```

## Step 3: Install ONNX Runtime with GPU Support

Download the appropriate ONNX Runtime GPU wheel file for your Python and JetPack versions from [Jetson Zoo](https://elinux.org/Jetson_Zoo#ONNX_Runtime). For Jetpack 5.1.2 and Python 3.8, we use ONNX Runtime version 1.18.0.

Install the wheel:

```bash
pip install onnxruntime_gpu-<version>.whl
```

## Step 4: Install GCC 11 and Additional CUDA Libraries for Compatibility

Add Ubuntu toolchain repository for GCC:

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-11 g++-11
```

Install necessary CUDA libraries:

```bash
sudo apt-get install libcublas-11-8 libcublas-dev-11-8
sudo apt-get install cuda-cudart-11-8
```

### Update Environment Variables for CUDA 11.8 Compatibility

Ensure compatibility by adding the following to your `~/.bashrc` (if not already present):

```bash
export PATH="/usr/local/cuda-11.8/bin:/home/<user>/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
```

Apply the changes again:

```bash
source ~/.bashrc
```

## Verification

Confirm installation by running:

```bash
nvcc --version
```

and verifying ONNX Runtime installation in Python:

```python
import onnxruntime
ort_device = ['CUDAExecutionProvider']
model_path = '<path_to_model>.onnx'
ort_session = onnxruntime.InferenceSession(model_path, providers=ort_device)
```

This setup should ensure correct installation and configuration of ONNX Runtime and CUDA on your Jetson Orin Nano device.

