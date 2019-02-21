import glob
import math
import numpy as np
import os
import fnmatch
import sys


def round_up(x, y):
    return int(math.ceil(float(x) / float(y)))


def add_parameter(class_object, kwargs, parameter, default=None):
    if parameter in kwargs:
        setattr(class_object, parameter, kwargs.get(parameter))
    else:
        setattr(class_object, parameter, default)


def additional_kwargs(class_object, kwargs):
    output_kwargs = {}
    for parameter in kwargs:
        if not hasattr(class_object, parameter):
            output_kwargs[parameter] = kwargs[parameter]

    return output_kwargs


def rot90(array, n=1, axis=2):

    """Rotate an array by 90 degrees in the counter-clockwise direction around the given axis
        Nabbed from https://stackoverflow.com/questions/33190042/how-to-calculate-all-24-rotations-of-3d-array
    """
    
    array = np.swapaxes(array, 2, axis)
    array = np.rot90(array, n)
    array = np.swapaxes(array, 2, axis)
    return array


def grab_files_recursive(input_directory, regex='*', return_dir=False, return_file=True, recursive=True):

    """ Returns all files recursively in a directory. Essentially a convenience wrapper 
        around os.walk.

        Parameters
        ----------

        input_directory: str
            The folder to search.
        regex: str
            A linux-style pattern to match.

        Returns
        -------
        output_list: list
            A list of found files.
    """

    output_list = []

    if recursive:

        for root, subFolders, files in os.walk(input_directory):
            if return_dir:
                for subFolder in subFolders:
                    if fnmatch.fnmatch(subFolder, regex):
                        output_list += [os.path.join(root, subFolder)]
            if return_file:
                for file in files:
                    if fnmatch.fnmatch(file, regex):
                        output_list += [os.path.join(root, file)]

    else:

        output_list = glob.glob(os.path.join(input_directory, regex))

    return output_list


def nifti_splitext(input_filepath):

    """ os.path.splitext splits a filename into the part before the LAST
        period and the part after the LAST period. This will screw one up
        if working with, say, .nii.gz files, which should be split at the
        FIRST period. This function performs an alternate version of splitext
        which does just that.

        TODO: Make work if someone includes a period in a folder name (ugh).

        Parameters
        ----------
        input_filepath: str
            The filepath to split.

        Returns
        -------
        split_filepath: list of str
            A two-item list, split at the first period in the filepath.

    """

    input_filepath = str(input_filepath)
    path_split = str.split(input_filepath, os.sep)
    basename = path_split[-1]
    split_filepath = str.split(basename, '.')

    if len(split_filepath) <= 1:
        return split_filepath
    else:
        return [os.path.join(os.sep.join(path_split[0:-1]), split_filepath[0]), '.' + '.'.join(split_filepath[1:])]


def replace_extension(input_filepath, extension):

    """Convenience function to safely switch out an extension.
    
    Parameters
    ----------
    input_filepath : str
        Filepath to switch extensions on.
    extension : str
        Extension to switch in.
    
    Returns
    -------
    str
        Filepath with switched extension.
    """

    input_filepath = os.path.abspath(input_filepath)
    basename = os.path.basename(input_filepath)
    pre_extension = str.split(basename, '.')[0]
    return os.path.join(os.path.dirname(input_filepath), pre_extension + extension)


def replace_suffix(input_filepath, input_suffix, output_suffix, suffix_delimiter=None, file_extension=None):

    """ Replaces an input_suffix in a filename with an output_suffix. Can be used
        to generate or remove suffixes by leaving one or the other option blank.

        Parameters
        ----------
        input_filepath: str
            The filename to be transformed.
        input_suffix: str
            The suffix to be replaced
        output_suffix: str
            The suffix to replace with.
        suffix_delimiter: str
            Optional, overrides input_suffix. Replaces whatever 
            comes after suffix_delimiter with output_suffix.

        Returns
        -------
        output_filepath: str
            The transformed filename
    """

    # Temp code to deal with directories, TODO
    if os.path.isdir(input_filepath):
        output_filepath = input_filepath + output_suffix + file_extension
        return output_filepath

    else:
        if '.' in os.path.basename(input_filepath):
            split_filename = nifti_splitext(input_filepath)
        else:
            split_filename = [input_filepath, '']

        if suffix_delimiter is not None:
            input_suffix = str.split(split_filename[0], suffix_delimiter)[-1]

        if input_suffix not in os.path.basename(input_filepath):
            print(('ERROR!', input_suffix, 'not in input_filepath.'))
            return []

        else:
            if input_suffix == '':
                prefix = split_filename[0]
            else:
                prefix = input_suffix.join(str.split(split_filename[0], input_suffix)[0:-1])
            prefix = prefix + output_suffix
            output_filepath = prefix + split_filename[1]

            if file_extension is not None:
                replace_extension(output_filepath, file_extension)

            return output_filepath


def make_dir(input_directory):

    """ Convenience function that adds os.path.exists to os.makedirs
    """

    if not os.path.exists(input_directory):
        os.makedirs(input_directory)


def quotes(input_string):

    """ Some command line function require filepaths with spaces in them to be in quotes.
    """

    return '"' + input_string + '"'


def cli_sanitize(input_filepath, save=False, delete=False):

    """ Copies out a filename without spaces, or deletes that file.
        Will not work for directories with spaces in their names.
    """

    input_filepath = os.path.abspath(input_filepath)
    new_filepath = os.path.join(os.path.dirname(input_filepath), os.path.basename(input_filepath).replace(' ', '__'))

    if delete:
        os.remove(new_filepath)
    if save:
        os.copy(input_filepath, new_filepath)

    return new_filepath


def docker_print(*args):

    """ Docker doesn't flush stdout in some circumstances, so one needs to do so manually.
    
    Parameters
    ----------
    *args
        Print parameters
    """

    print(args)
    sys.stdout.flush()