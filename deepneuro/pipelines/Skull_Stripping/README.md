# Segment_GBM

This module creates segmentations of brain masks given pre-contrast T1 and FLAIR input volumes. These segmentations are created by deep neural networks trained on hundreds of public and private datasets of pre-operative high- and low-grade GBMs. The following pre-processing steps are included in module: N4Bias Correction, Isotropic Resampling (1x1x1), Image Registration, and Zero-Mean normalization. This module was developed at the Quantitative Tumor Imaging Lab at the Martinos Center (MGH, MIT/Harvard HST).

## Table of Contents
- [Docker Usage](#docker-usage)
- [Python Docker Wrapper Usage](#python-docker-wrapper-usage)
- [Docker Example](#docker-example)

## Docker Usage

The best way to use this module is with a Docker container. If you are not familiar with Docker, you can download it [here](https://docs.docker.com/engine/installation/) and read a tutorial on proper usage [here](https://docker-curriculum.com/).

Pull the Skull_Stripping Docker container from https://hub.docker.com/r/qtimlab/deepneuro_segment_gbm/. Use the command "docker pull qtimlab/deepneuro_segment_gbm".

You can then create a command using the following template to create a glioblastoma segmentation:

```
nvidia-docker run --rm -v [MOUNTED_DIRECTORY]:/INPUT_DATA qtimlab/deepneuro_segment_gbm skull_stripping pipeline -T1POST <file> -FLAIR <file> -output_folder <directory> [-gpu_num <int> -debiased -resampled -registered -save_all_steps -save_preprocessed]
```

In order to use Docker, you must mount the directory containing all of your data and your output. All inputted filepaths must be relative to this mounted directory. For example, if you mounted the directory /home/my_users/data/, and wanted to input the file /home/my_users/data/patient_1/FLAIR.nii.gz as a parameter, you should input /INPUT_DATA/patient_1/FLAIR.nii.gz. Note that the Python wrapper for Docker in this module will adjust paths for you.

A brief explanation of this functions parameters follows:

| Parameter       | Documenation           |
| ------------- |-------------|
| -output_folder | A filepath to your output folder. Two nifti files will be generated "enhancingtumor.nii.gz" and "wholetumor.nii.gz" |
| -T1, -T1POST, -FLAIR      | Filepaths to input MR modalities. Inputs can be either nifti files or DICOM folders. Note that DICOM folders should only contain one volume each.      |
| -wholetumor_output, -enhancing_output | Optional. Name of output for wholetumor and enhancing labels, respectively. Should not be a filepath, like '/home/user/enhancing.nii.gz', but just a name, like "enhancing"      |
| -gpu_num | Optional. Which CUDA GPU ID # to use. Defaults to 0, i.e. the first gpu. |
| -debiased | If flagged, data is assumed to already have been N4 bias-corrected, and skips that preprocessing step. |
| -resampled | If flagged, data is assumed to already have been isotropically resampled, and skips that preprocessing step. |
| -registered | If flagged, data is assumed to already have been registered into the same space, and skips that preprocessing step. |
| -skullstripped | If flagged, data is assumed to already have been skullstripped, and skips that preprocessing step. |
| -save_all_steps | If flagged, intermediate volumes in between preprocessing steps will be saved in output_folder. |
| -save_preprocessed | If flagged, the final volumes after bias correction, resampling, and registration. |

## Python Docker Wrapper Usage

To avoid adjusting your  you may want to avoid using nvidia-docker directly. I've also created a python utility that wraps around the nvidia-docker command above, and is slightly easier to use. In order to use this utlity, you will need to clone this repository. ("git clone https://github.com/QTIM-Lab/DeepNeuro"), and install it ("python setup.py install", in the directory you cloned the repository).

Once you have installed the repository, you can use the following command on the command-line:

```
segment_gbm docker_pipeline -T1 <file> -T1POST <file> -FLAIR <file> -output_folder <directory> [-gpu_num <int> -bias -resampled -registered -save_all_steps -save_preprocessed
```

Parameters should be exactly the same as in the Docker use-case, except now you will not have to modify filepaths to be relative to the mounted folder.

## Docker Example

Let's say you stored some DICOM data on your computer at the path /home/my_user/Data/, and wanted to segment data located at /home/my_user/Data/Patient_1. The nvidia-docker command would look like this:

```
nvidia-docker run --rm -v /home/my_user/Data:/INPUT_DATA qtimlab/deepneuro_segment_gbm segment_gbm pipeline -T1 /INPUT_DATA/Patient_1/T1pre -T1POST /INPUT_DATA/Patient_1/T1post -FLAIR /INPUT_DATA/Patient_1/FLAIR -output_folder /INPUT_DATA/Patient_1/Output_Folder
```

First, note that the "/INPUT_DATA" designation on the right-hand side of the "-v" option will never change. "INPUT_DATA" is a folder within the Docker container that will not change between runs.

Second, note that you will need to make sure that the left-hand side of the "-v" option is an absolute, rather than relative, path. For example "../Data/" and "~/Data/" will not work (relative path), but "/home/my_user/Data/" will work (absolute path, starting from the root directory).

Third, note that the folders you provide as arguments to the "segment pipeline" command should be relative paths. This is because you are mounting, and thus renaming, a folder on your system to the "/INPUT_DATA" folder inside the Docker system. For example, if you were mounting the directory "/home/my_user/Data/" to "/INPUT_DATA", you should not provide the path "/home/my_user/Data/Patient_1/FLAIR" as a parameter. Rather, you should provide the path "/INPUT_DATA/Patient_1/FLAIR", as those parts of the path are within the scope of your mounted directory.
