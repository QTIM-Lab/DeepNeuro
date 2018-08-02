import os
import glob

from collections import defaultdict


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


def parse_data_directories(data_collection, modality_dict, case_list=None, recursive=True):

    """ Recursive functionality not yet available
    """

    # If a predefined case list is provided, only choose these cases.
    if data_collection.case_list is not None and os.path.basename(subject_dir) not in data_collection.case_list:
        return

    lead_group = modality_dict[modality_dict.keys()[0]]
    lead_files = []

    for directory in lead_group:
        if os.path.isdir(os.path.normpath(directory)):
            directory = os.path.join(directory, '*')

        lead_files += grab_files

    for filepath in lead_group

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