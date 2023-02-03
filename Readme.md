Organizing AI | ML Stuff in here
			 😁
			 
Setting up Linux for AI | ML | DL 👇

### A guide on how to setup ML environment on Linux

## 1. Install Anaconda or MiniConda

Download Anaconda from [here](https://www.anaconda.com/products/individual#Downloads) and install it.

## 2. Create a new environment

```bash
conda create -n ml python=3.10.9
```

## 3. Activate the environment

```bash
conda activate ml
```

## 4. Install packages 
    ```bash
    conda install --file requirements.txt
    ```
    
    OR
    
    
    ```bash
    conda install numpy
    conda install pandas
    conda install matplotlib
    conda install seaborn
    conda install scikit-learn
    conda install jupyter
    conda install scikit-learn-intelex # Intel optimized scikit-learn, only for Intel CPUs     
    ```

## 5. Conda install from requirements.txt

```bash
conda install --file requirements.txt
```

## 6. Getting CUDA tollkit

Go to [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) and download the latest version of CUDA toolkit.
Following Command will be different for different versions of CUDA toolkit.

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.85.12_linux.run

sudo sh cuda_12.0.1_525.85.12_linux.run
```
This will take some time to download as it is ~3Gb.

Select Driver if you don't have it installed already,else deselect it.

## 7. Editing Startup file
Edit /etc/profile and add the following lines at the end of the file.
```
# Vars for nvidia things
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
```

## 8. Install cuDNN
Download cuDNN from [here](https://developer.nvidia.com/rdp/cudnn-archive) .
Do
```
tar -xvf cudnn-11.xyz.tgz
```
and copy the files to the CUDA Toolkit directory.
```
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64 #this removes symlinks , so tar and untar again. To do
cd lib/
tar czvf lib.tgz *

sudo cp lib.tgz /usr/local/cuda/lib64

cd /usr/local/cuda/lib64

tar -xvf lib.tgz
```

## 9. Install Tensorrt
Just pip it
```
pip install tensorrt
```
