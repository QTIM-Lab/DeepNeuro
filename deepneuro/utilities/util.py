import math
import os
import fnmatch

def round_up(x, y):
    return int(math.ceil(float(x) / float(y)))


def add_parameter(class_object, kwargs, parameter, default=None):
    if parameter in kwargs:
        setattr(class_object, parameter, kwargs.get(parameter))
    else:
        setattr(class_object, parameter, default)


def grab_files_recursive(input_directory, regex='*'):

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

    for root, subFolders, files in os.walk(input_directory):
        for file in files:
            if fnmatch.fnmatch(file, regex):
                output_list += [os.path.join(root, file)]

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

    path_split = str.split(input_filepath, os.sep)
    basename = path_split[-1]
    split_filepath = str.split(basename, '.')

    if len(split_filepath) <= 1:
        return split_filepath
    else:
        return [os.path.join(os.sep.join(path_split[0:-1]), split_filepath[0]), '.' + '.'.join(split_filepath[1:])]

def replace_suffix(input_filepath, input_suffix, output_suffix, suffix_delimiter=None, file_extension='.nii.gz'):

    """ Replaces an input_suffix in a filename with an output_suffix. Can be used
        to generate or remove suffixes by leaving one or the other option blank.

        TODO: Make suffixes accept regexes. Can likely replace suffix_delimiter after this.
        TODO: Decide whether suffixes should extend across multiple directory levels.

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
        split_filename = nifti_splitext(input_filepath)

        if suffix_delimiter is not None:
            input_suffix = str.split(split_filename[0], suffix_delimiter)[-1]

        if input_suffix not in os.path.basename(input_filepath):
            print 'ERROR!', input_suffix, 'not in input_filepath.'
            return []

        else:
            if input_suffix == '':
                prefix = split_filename[0]
            else:
                prefix = input_suffix.join(str.split(split_filename[0], input_suffix)[0:-1])
            prefix = prefix + output_suffix
            output_filepath = prefix + split_filename[1]

            if file_extension is not None:
                if not output_filepath.endswith(file_extension):
                    output_filepath = output_filepath + file_extension

            return output_filepath