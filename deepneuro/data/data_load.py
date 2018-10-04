import os
import glob
import csv
import tables

from deepneuro.utilities.util import grab_files_recursive, nifti_splitext


def parse_directories(data_collection, data_directories, case_list=None):

    """ Broken out from fill_data_groups.
    """

    if data_collection.verbose:
        print(('Gathering image data from...', data_directories, '\n'))

    # Iterate through directories.. Always looking for a better way to check optional list typing.
    directory_list = []
    for d in data_directories:
        if not os.path.exists(d):
            print(('WARNING: One of the data directories you have input,', d, 'does not exist!'))
        directory_list += glob.glob(os.path.join(d, "*/"))
    directory_list = sorted(directory_list)

    for data_directory, data_groups in data_directories.items():

        directory_list = glob.glob(os.path.join(data_directory, '*/'))

        for directory in directory_list:

            case_name = os.path.abspath(directory)

            for data_group, sequence_labels in data_groups.items():

                data_group_files = []
                for sequence in sequence_labels:

                    # Iterate through patterns.. Always looking for a better way to check optional list typing.
                    if isinstance(sequence, str):
                        target_file = glob.glob(os.path.join(case_name, sequence))
                    else:
                        target_file = []
                        for m in sequence:
                            target_file += glob.glob(os.path.join(case_name, m))

                    if len(target_file) == 1:
                        data_group_files.append(target_file[0])
                    else:
                        print(('Error loading', sequence, 'from', os.path.basename(os.path.dirname(case_name))))
                        if len(target_file) == 0:
                            print('No file found.\n')
                        else:
                            print('Multiple files found.\n')
                        return

                if len(data_group_files) == len(sequence_labels):
                    data_collection.data_groups[data_group].add_case(os.path.abspath(case_name), list(data_group_files))

            data_collection.cases.append(case_name)


def parse_hdf5(data_collection, data_hdf5, case_list=None):

    """ TODO: Add support for loading in multiple hdf5s
        TODO: Keep track of open hdf5 file.
    """

    if data_collection.verbose:
        print('Gathering image data from...')

    open_hdf5 = tables.open_file(data_hdf5, "r")

    for data_group in open_hdf5.root._f_iter_nodes():
        if '_affines' not in data_group.name and '_casenames' not in data_group.name:
            
            # Affines and Casenames. Also not great praxis.
            data_collection.data_groups[data_group.name].data_affines = getattr(open_hdf5.root, data_group.name + '_affines')
            data_collection.data_groups[data_group.name].data_casenames = getattr(open_hdf5.root, data_group.name + '_casenames')

            # There's some double-counting here. TODO: revise, chop down one or the other.
            data_collection.data_groups[data_group.name].cases = list(range(data_group.shape[0]))
            data_collection.data_groups[data_group.name].case_num = data_group.shape[0]
            data_collection.cases = list(range(data_group.shape[0]))


def parse_filepaths(data_collection, data_group_dict, case_list=None, recursive=True, verbose=True, file_identifying_chars=None):

    """ Recursive functionality not yet available
    """

    # Case lists not yet implemented.

    # Pulling from multiple directories not yet implemented.
    lead_group = data_group_dict[list(data_group_dict.keys())[0]]

    if type(lead_group[0]) is list:
        lead_directory = os.path.abspath(lead_group[0][0])
    else:
        lead_directory = os.path.abspath(lead_group[0])

    lead_files = []

    for directory in lead_group:

        if os.path.isdir(os.path.normpath(directory)):
            directory = os.path.join(directory, '*')

        regex = os.path.basename(directory)

        lead_files += grab_files_recursive(os.path.abspath(os.path.dirname(directory)), regex=regex, recursive=recursive)

    for lead_filepath in lead_files:

        base_filedir = os.path.dirname(lead_filepath).split(lead_directory, 1)[1]
        base_filepath = nifti_splitext(lead_filepath)[0]

        if file_identifying_chars is not None:
            base_filepath = os.path.basename(os.path.join(os.path.dirname(base_filepath), os.path.basename(base_filepath)[:file_identifying_chars]))
        
        # Search for sequence files, and skip those missing with files modalities.
        for data_group, sequence_labels in list(data_collection.data_group_dict.items()):

            data_group_files = []

            for sequence in sequence_labels:

                target_file = glob.glob(os.path.join(sequence, base_filedir, base_filepath + '*'))

                if len(target_file) == 1:
                    data_group_files.append(target_file[0])
                else:
                    print(('Error loading', sequence, 'from case', lead_filepath))
                    if len(target_file) == 0:
                        print('No file found.\n')
                    else:
                        print('Multiple files found.\n')

            if len(data_group_files) == len(sequence_labels):
                data_collection.data_groups[data_group].add_case(lead_filepath, data_group_files)
            else:
                lead_filepath = None
                break

        # This is ugh.
        if lead_filepath is not None:
            case_name = lead_filepath
            data_collection.cases.append(case_name)


def parse_csv(data_collection, data_csv, case_list=None):

    input_csvs = set()

    if type(data_csv) is str:
        input_csvs = [data_csv]
    else:
        for data_group_name, csv_file in list(data_csv.items()):
            input_csvs.update([csv_file])

    for input_csv in input_csvs:
        with open(input_csv, 'r') as infile:
            csv_reader = csv.reader(infile)
            header = next(csv_reader)
            data_group_names = set(header[1:])

            for row in csv_reader:

                casename = row[0]
                data_group_files = {name: [] for name in data_group_names}

                for idx, data_group_name in enumerate(header[1:]):

                    data_group_files[data_group_name] += [row[idx + 1]]

                for data_group_name in data_group_names:    
                    data_collection.data_groups[data_group_name].add_case(casename, data_group_files[data_group_name])

                data_collection.cases.append(casename)

    return


def parse_numpy(data_collection, data_numpy, case_list=None):

    raise NotImplementedError
    

if __name__ == '__main__':

    pass