# Segment_GBM

This command-line module creates segmentations of "whole tumor" (edema + contrast-enhancing + necrosis) and contrasting-enhancing tumor given pre-contrast T1, post-contrast T1, and FLAIR input volumes. These segmentations are created by deep neural networks trained on hundreds of public and private datasets of post-operative GBMs. The following pre-processing steps are included in module: N4Bias Correction (3DSlicer), Image Registration (3DSlicer), Skull-Stripping (DeepNeuro), and Zero-Mean normalization. This module can take in as input NIfTI files (.nii.gz) and DICOM directories (/\*.dcm). This module was developed at the Quantitative Tumor Imaging Lab at the Martinos Center (MGH, MIT/Harvard HST).

## Table of Contents
- [Local Command-Line Usage](#command-line-usage)
- [Docker Usage](#docker-usage)
- [Singularity Usage](#singularity-usage)
- [Python Wrapper Usage](#python-wrapper-usage)
- [Docker and Singularity Example](#docker-and-singularity-example)
- [Citation](#citation)

## Command Line Usage

If you have installed DeepNeuro locally on your workstation, without the use of Docker or Singularity, you can run this module directly from the command line. The basic format of the command is as follows:

```
segment_gbm pipeline -T1 <file> -T1POST <file> -FLAIR <file> -output_folder <directory> -wholetumor_output <string> -enhancing_output <string> [-debiased -registered -skullstripped -preprocessed -gpu_num <int> -save_all_steps -save_only_segmentations -quiet]
```

This functions basic parameters are as follows:

| Parameter       | Documenation           |
| ------------- |-------------|
| -output_folder | A filepath to your output folder. Two nifti files will be generated "enhancingtumor.nii.gz" and "wholetumor.nii.gz" |
| -T1, -T1POST, -FLAIR      | Filepaths to input MR modalities. Inputs can be either nifti files or DICOM folders. Note that DICOM folders should only contain one volume each.      |
| -wholetumor_output, -enhancing_output | Optional. Name of output filepaths for wholetumor and enhancing labels, respectively. Should not be a filepath, like '/home/user/enhancing.nii.gz', but just a name, like "enhancing.nii.gz". Files with these names will be output into your output_folder.      |

This DeepNeuro pipeline assumes that your data will need to be preprocessed before it is preprocessed. However, you may have already performed some of these preprocessing steps yourself. You can skip some preprocessing steps by adding the following flags to your command:

| Parameter       | Documenation           |
| ------------- |-------------|
| -debiased | If flagged, data is assumed to already have been N4 bias-corrected, and skips that preprocessing step. |
| -registered | If flagged, data is assumed to already have been registered into the same space, and skips that preprocessing step. |
| -skullstripped | If flagged, data is assumed to already have been skullstripped, and skips that preprocessing step. DeepNeuro assumes that skullstripped data has had non-brain tissue replaced with the intensity value '0'. |
| -preprocessed | If flagged, data is assumed to already have been entirely preprocessed by DeepNeuro, including intensity normalization. Only use if data has been passed through DeepNeuro already to ensure reproducible performance. |

You can also turn on additional miscellaneous parameters with the following flags:

| Parameter       | Documenation           |
| ------------- |-------------|
| -gpu_num | Optional. Which CUDA GPU ID # to use, if your workstation has multiple GPUs. Defaults to 0, i.e. the first gpu. |
| -save_all_steps | If flagged, input volumes will be saved out after each preprocessing step, allowing to evaluate the differences between raw and pre-processed data. |
| -save_only_segmentations | By default, this module will output preprocessed data volumes in addition to segmentations. Turn this flag on in order to ONLY output segmentations. |
| -quiet | If flagged, this module will run in quiet mode, with no command-line output. |

In order to run this command-line locally, you must have installed and added to your path 3DSlicer, and have it added to your workstation's system path. If you are not able to install 3DSlicer, or do not have the technical skills to add it to you system path, we advise you use either a Docker or Singularity container as detailed below.

## Docker Usage

The easiest way to use this module is with a Docker container. If you are not familiar with Docker, you can download it [here](https://docs.docker.com/engine/installation/) and read a tutorial on proper usage [here](https://docker-curriculum.com/).

Before you can use this container, you must first pull the Segment_GBM Docker container from https://hub.docker.com/r/qtimlab/deepneuro_segment_gbm/. Use the command ```docker pull qtimlab/deepneuro_segment_gbm```.

You can then create a command for this module using the following template:

```
nvidia-docker run --rm -v [MOUNTED_DIRECTORY]:/INPUT_DATA qtimlab/deepneuro_segment_gbm segment_gbm pipeline -T1 <file> -T1POST <file> -FLAIR <file> -output_folder <directory> -wholetumor_output <string> -enhancing_output <string> [-debiased -registered -skullstripped -preprocessed -gpu_num <int> -save_all_steps -save_only_segmentations -quiet]
```

Notice that the parameters used here are the same parameters used in the section above, [Command Line Usage](#command-line-usage). Please refer to this section for details on each of this commands parameters.

In order to use Docker, you must mount the directory containing all of your data and your output. All filepaths input to DeepNeuro must be relative to this mounted directory. For example, if you mounted the directory /home/my_users/data/, and wanted to input the file /home/my_users/data/patient_1/FLAIR.nii.gz as a parameter, you should input /INPUT_DATA/patient_1/FLAIR.nii.gz. For more detail on how to mount a directory, see [Docker Example](docker-example)

## Singularity Usage

Singularity is a software that operates very similarly to Docker, but is more preferred in certain shared computing cluster environments. If you are not familiar with Singularity, you can install it [here](https://singularity.lbl.gov/docs-installation) and read a tutorial on proper usage [here](https://singularity.lbl.gov/quickstart).

You can then create a command for this module using the following template:

```
singularity exec --nv docker://qtimlab/deepneuro_segment_gbm -B [MOUNTED_DIRECTORY]:/INPUT_DATA segment_gbm pipeline -T1 <file> -T1POST <file> -FLAIR <file> -output_folder <directory> -wholetumor_output <string> -enhancing_output <string> [-debiased -registered -skullstripped -preprocessed -gpu_num <int> -save_all_steps -save_only_segmentations -quiet]
```

Notice that the parameters used here are the same parameters used in the section above, [Command Line Usage](#command-line-usage). Please refer to this section for details on each of this commands parameters.

In order to use Singularity, you must mount the directory containing all of your data and your output. All filepaths input to DeepNeuro must be relative to this mounted directory. For example, if you mounted the directory /home/my_users/data/, and wanted to input the file /home/my_users/data/patient_1/FLAIR.nii.gz as a parameter, you should input /INPUT_DATA/patient_1/FLAIR.nii.gz. For more detail on how to mount a directory, see [Docker Example](docker-example)

## Python Wrapper Usage

If you don't want to type out Docker/Singularity commands directly, don't want to avoid figuring out precisely which drives to mount, or want to integrate DeepNeuro commands into a Python processing script, you can also easily run DeepNeuro containers from a Python script. also created a python utility that wraps around the nvidia-docker command above, and is slightly easier to use. 

You will need to install DeepNeuro, or be running a DeepNeuro Singularity/Docker container, before you can use this command. Once you have installed the repository, you can use the following command structure to use this module:

```
from deepneuro.pipelines import predict_GBM

predict_GBM(output_folder, 
                T1POST, 
                FLAIR, 
                T1PRE, 
                bias_corrected=False, 
                resampled=False, 
                registered=False, 
                skullstripped=False, 
                preprocessed=False, 
                save_only_segmentations=False, 
                save_all_steps=False, 
                output_wholetumor_filename='wholetumor_segmentation.nii.gz', 
                output_enhancing_filename='enhancing_segmentation.nii.gz',
                quiet=False)
```

The parameters in this Python function should correspond to the parameters for the Docker/Singularity containers above.

## Docker and Singularity Example

Let's say you stored some DICOM data on your computer at the path /home/my_user/Data/, and wanted to segment data located at /home/my_user/Data/Patient_1. The commands would look like this:

```
nvidia-docker run --rm -v /home/my_user/Data:/INPUT_DATA qtimlab/deepneuro_segment_gbm segment_gbm pipeline -T1 /INPUT_DATA/Patient_1/T1pre -T1POST /INPUT_DATA/Patient_1/T1post -FLAIR /INPUT_DATA/Patient_1/FLAIR -output_folder /INPUT_DATA/Patient_1/Output_Folder

singularity exec --nv docker://qtimlab/deepneuro_segment_gbm -B /home/my_user/Data:/INPUT_DATA qtimlab/deepneuro_segment_gbm segment_gbm pipeline -T1 /INPUT_DATA/Patient_1/T1pre -T1POST /INPUT_DATA/Patient_1/T1post -FLAIR /INPUT_DATA/Patient_1/FLAIR -output_folder /INPUT_DATA/Patient_1/Output_Folder
```

First, note that the "/INPUT_DATA" designation on the right-hand side of the "-v" option will never change. "INPUT_DATA" is a folder within the Docker container that will not change between runs.

Second, note that you will need to make sure that the left-hand side of the "-v" option is an absolute, rather than relative, path. For example "../Data/" and "~/Data/" will not work (relative path), but "/home/my_user/Data/" will work (absolute path, starting from the root directory).

Third, note that the folders you provide as arguments to the "segment_gbm pipeline" command should be relative paths. This is because you are mounting, and thus renaming, a folder on your system to the "/INPUT_DATA" folder inside the Docker system. For example, if you were mounting the directory "/home/my_user/Data/" to "/INPUT_DATA", you should not provide the path "/home/my_user/Data/Patient_1/FLAIR" as a parameter. Rather, you should provide the path "/INPUT_DATA/Patient_1/FLAIR", as those parts of the path are within the scope of your mounted directory.

## Citation

@article{beers2017sequential,
  title={Sequential 3D U-Nets for Biologically-Informed Brain Tumor Segmentation},
  author={Beers, Andrew and Chang, Ken and Brown, James and Sartor, Emmett and Mammen, CP and Gerstner, Elizabeth and Rosen, Bruce and Kalpathy-Cramer, Jayashree},
  journal={arXiv preprint arXiv:1709.02967},
  year={2017}
}
