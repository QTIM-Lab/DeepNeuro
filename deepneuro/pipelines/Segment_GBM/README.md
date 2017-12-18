If you are experienced with Docker, you may want to use DeepNeuro directly from its Docker container without downloading the package. Each qtim_gbmSegmenter command will take the same basic format:

```
nvidia-docker run --rm -v [MOUNTED_DIRECTORY]:/INPUT_DATA qtimlab/qtim-gbmsegmenter segment docker_pipeline <T2> <T1pre> <T1post> <FLAIR> <output_folder> [-gpu_num <int> -niftis -nobias -preprocessed -keep_outputs]
```

All input folders (T2_Folder, Output_Folder, etc.) are presumed to be located in MOUNTED_DIRECTORY. Note that the filepaths you provide in the parameters for this function have to be specifally formatted to be launched from nvidia-docker -- see more details in the Example section below.

An brief explanation of parameters after docker_pipeline follows.

* T2, T1pre, T1post, FLAIR: Filepaths to DICOM folders. Can be filepaths to niftis if the -niftis flag is set.
* output_folder: A filepath to your output folder. This folder will be created if it does not already exist.
* -gpu_num: Which CUDA GPU ID # to use.
* -niftis: Input nifti files instead of DIOCM folders.
* -nobias: Skip the bias correction step.
* -preprocessed: Skip bias correction, resampling, and registration.
* -keep_outputs: Do not delete files generated from intermediary steps.

### Example

Let's say you stored some DICOM data on your computer at the path /home/my_user/Data/, and wanted to segment data located at /home/my_user/Data/Patient_1. The nvidia-docker command would look like this:

```
nvidia-docker run --rm -v /home/my_user/Data:/INPUT_DATA qtimlab/qtim-gbmsegmenter segment pipeline Patient_1/T2 Patient_1/T1pre Patient_1/T1post Patient_1/FLAIR Patient_1/Output_Folder
```

First, note that the "/INPUT_DATA" designation on the right-hand side of the "-v" option will never change. "INPUT_DATA" is a folder within the Docker container that will not change between runs.

Second, note that you will need to make sure that the left-hand side of the "-v" option is an absolute, rather than relative, path. For example "../Data/" and "~/Data/" will not work (relative path), but "/home/my_user/Data/" will work (absolute path, starting from the root directory).

Third, note that the folders you provide as arguments to the "segment pipeline" command should be relative paths. This is because you are mounting, and thus renaming, a folder on your system to the "/INPUT_DATA" folder inside the Docker system. For example, if you were mounting the directory "/home/my_user/Data/" to "/INPUT_DATA", you should not provide the path "/home/my_user/Data/Patient_1/T2" as a parameter. Rather, you should provide the path "Patient_1/T2", as those parts of the path are within the scope of your mounted directory.

## Python Wrapper Usage

Given the filepath rules above, you may want to avoid using nvidia-docker directly. I've also created a python utility that wraps around the nvidia-docker command above, and is slightly easier to use. In order to use this utlity, you will need to clone this repository. ("git clone https://github.com/QTIM-Lab/qtim_gbmSegmenter"), and install it ("python setup.py install", in the directory you cloned the repository).

Once you have installed the repository, you can use the following command on the command-line:

```
segment pipeline <T2> <T1pre> <T1post> <FLAIR> <output_folder> [-gpu_num <int> -niftis -nobias -preprocessed -keep_outputs]
```

Here are some details about what each of those parameters mean.

* T2, T1pre, T1post, FLAIR: Filepaths to DICOM folders. Can be filepaths to niftis if the -niftis flag is set.
* output_folder: A filepath to your output folder. This folder will be created if it does not already exist.
* -gpu_num: Which CUDA GPU ID # to use.
* -niftis: Input nifti files instead of DICOM folders.
* -nobias: Skip the bias correction step.
* -preprocessed: Skip bias correction, resampling, and registration.
* -keep_outputs: Do not delete files generated from intermediary steps.
