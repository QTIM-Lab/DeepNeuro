import os
import glob

from collections import defaultdict

from deepneuro.utilities.util import grab_files_recursive, nifti_splitext


def parse_subject_directory(data_collection, subject_dir, case_list=None):

    """ Broken out from fill_data_groups.
    """

    # If a predefined case list is provided, only choose these cases.
    if data_collection.case_list is not None and os.path.basename(subject_dir) not in data_collection.case_list:
        return

    # Search for modality files, and skip those missing with files modalities.
    for data_group, modality_labels in data_collection.modality_dict.items():

        modality_group_files = []
        for modality in modality_labels:

            # Iterate through patterns.. Always looking for a better way to check optional list typing.
            if isinstance(modality, str):
                target_file = glob.glob(os.path.join(subject_dir, modality))
            else:
                target_file = []
                for m in modality:
                    target_file += glob.glob(os.path.join(subject_dir, m))

            if len(target_file) == 1:
                modality_group_files.append(target_file[0])
            else:
                print('Error loading', modality, 'from', os.path.basename(os.path.dirname(subject_dir)))
                if len(target_file) == 0:
                    print('No file found.\n')
                else:
                    print('Multiple files found.\n')
                return

        if len(modality_group_files) == len(modality_labels):
            data_collection.data_groups[data_group].add_case(os.path.abspath(subject_dir), list(modality_group_files))

    case_name = os.path.abspath(subject_dir)
    data_collection.cases.append(case_name)
    data_collection.preprocessed_cases[case_name] = defaultdict(list)


def parse_modality_directories(data_collection, modality_dict, case_list=None, recursive=True, verbose=True, identifying_chars=None):

    """ Recursive functionality not yet available
    """

    # Cases not yet implemented.

    # Pulling from multiple directories not yet implemented.
    lead_group = modality_dict[modality_dict.keys()[0]]
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

        if identifying_chars is not None:
            base_filepath = os.path.basename(os.path.join(os.path.dirname(base_filepath), os.path.basename(base_filepath)[:identifying_chars]))
        
        # Search for modality files, and skip those missing with files modalities.
        for data_group, modality_labels in data_collection.modality_dict.items():

            modality_group_files = []

            for modality in modality_labels:

                target_file = glob.glob(os.path.join(modality, base_filedir, base_filepath + '*'))

                if len(target_file) == 1:
                    modality_group_files.append(target_file[0])
                else:
                    print('Error loading', modality, 'from case', lead_filepath)
                    if len(target_file) == 0:
                        print('No file found.\n')
                    else:
                        print('Multiple files found.\n')

            if len(modality_group_files) == len(modality_labels):
                data_collection.data_groups[data_group].add_case(lead_filepath, modality_group_files)
            else:
                lead_filepath = None
                break

        # This is ugh.
        if lead_filepath is not None:
            case_name = lead_filepath
            data_collection.cases.append(case_name)
            data_collection.preprocessed_cases[case_name] = defaultdict(list)