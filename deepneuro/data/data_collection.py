from __future__ import division

import os
import glob

import numpy as np
import tables
import nibabel as nib

from image_utils import nifti_2_numpy
from deepneuro.augmentation.augment import Augmentation, AugmentationGroup

class DataCollection(object):

    def __init__(self, data_directory, modality_dict, spreadsheet_dict=None, value_dict=None, case_list=None, augmentations=None, brainmask_dir=None, roimask_dir=None, patch_shape=None):

        # Input vars
        self.data_directory = os.path.abspath(data_directory)
        self.modality_dict = modality_dict
        self.spreadsheet_dict = spreadsheet_dict
        self.value_dict = value_dict
        self.case_list = case_list

        # Special behavior for augmentations
        self.augmentations = []
        self.cases = []

        # Empty vars
        self.data_groups = {}
        self.data_shape = None
        self.data_shape_augment = None

        # TEMPORARY
        self.brainmask_dir = brainmask_dir
        self.roimask_dir = roimask_dir

        # Data-Checking, e.g. duplicate data keys, data_shape
        self.patch_shape = patch_shape

    def verify_data_shape(self):
        return

    def append_augmentation(self, augmentation_group, data_groups=None):

        for data_group_label in augmentation_group.augmentation_dict.keys():
            self.data_groups[data_group_label].append_augmentation(augmentation_group.augmentation_dict[data_group_label])

            augmentation_group.augmentation_dict[data_group_label].append_data_group(self.data_groups[data_group_label])
            augmentation_group.augmentation_dict[data_group_label].initialize_augmentation()

        # Don't like this list method, find a more explicity way.
        self.augmentations.append(augmentation_group)

        return

    def fill_data_groups(self):

        print 'Gathering image data from...', self.data_directory

        #####
        # TODO: Add section for spreadsheets.
        # TODO: Add section for values.
        #####

        # Imaging modalities added.

        for modality_group in self.modality_dict:

            # print modality_group
            if modality_group not in self.data_groups.keys():
                self.data_groups[modality_group] = DataGroup(modality_group)

        for subject_dir in sorted(glob.glob(os.path.join(self.data_directory, "*/"))):

            self.cases.append(os.path.abspath(subject_dir))

            for modality_group in self.modality_dict:

                if self.case_list is not None:
                    if os.path.basename(subject_dir) not in self.case_list:
                        continue

                modality_group_files = []

                for modality in self.modality_dict[modality_group]:
                    target_file = glob.glob(os.path.join(subject_dir, modality))
                    try:
                        modality_group_files.append(target_file[0])
                    except:
                        print 'Error loading', modality, 'from', os.path.basename(os.path.dirname(subject_dir))
                        break

                if len(modality_group_files) == len(self.modality_dict[modality_group]):
                    self.data_groups[modality_group].add_case(tuple(modality_group_files), os.path.abspath(subject_dir))

    def append_data_to_file(self, input_hdf5, data_groups=None, save_masks=False, store_masks=True):

        if data_groups is None:
            data_groups = self.data_groups.keys()

        if output_filepath is None:
            output_filepath = os.path.join(self.data_directory, 'data.hdf5')

        # Create Data File
        try:
            hdf5_file = create_hdf5_file(output_filepath, data_groups, self, save_masks, store_masks)
        except Exception as e:
            os.remove(output_filepath)
            raise e

        self.write_image_data_to_storage(data_groups, save_masks, store_masks)

        hdf5_file.close()

        return

    def write_data_to_file(self, output_filepath=None, data_groups=None, save_masks=False, store_masks=True):

        """ Interesting question: Should all passed data_groups be assumed to have equal size? Nothing about hdf5 requires that, but it makes things a lot easier to assume.

            TODO: Integrate mask code into whole DataCollection object.
        """


        if data_groups is None:
            data_groups = self.data_groups.keys()

        if output_filepath is None:
            output_filepath = os.path.join(self.data_directory, 'data.hdf5')

        # Create Data File
        try:
            hdf5_file = create_hdf5_file(output_filepath, data_groups, self, save_masks, store_masks)
        except Exception as e:
            os.remove(output_filepath)
            raise e

        # Write data
        self.write_image_data_to_storage(data_groups, save_masks, store_masks)

        hdf5_file.close()

    def write_image_data_to_storage(self, data_groups, save_masks=False, store_masks=True):

        """ Some of the syntax around data groups can be cleaned up in this function.
        """

        # Write data
        data_group_objects = [self.data_groups[label] for label in data_groups]

        # Will cases always be ordered..?
        for case_idx, case_name in enumerate(self.cases):

            print 'Working on image.. ', case_idx, 'in', case_name

            all_cases = {}

            # Check for missing data.
            missing_case = False
            for data_group in data_group_objects:

                try:
                    case_index = data_group.cases.index(case_name)
                except:
                    missing_case = True
                    missing_data_group = data_group.label
                    break

                data_group.base_case, data_group.base_affine = read_image_files(data_group.data[case_index], return_affine=True)
                data_group.base_case = data_group.base_case[:][np.newaxis]
                data_group.base_casename = case_name
                data_group.current_case = data_group.base_case

            if missing_case:
                print 'Missing case', case_name, 'in data group', missing_data_group, '. Skipping this case..'
                continue

            print '\n'

            self.recursive_augmentation(data_group_objects, 0, save_masks, store_masks)


    def recursive_augmentation(self, data_groups, augmentation_num=0):

        """ This function baldly reveals my newness at recursion..
        """
        # Write data
        print 'BEGIN RECURSION FOR AUGMENTATION NUM', augmentation_num

        if augmentation_num == len(self.augmentations):

            for data_group in data_groups:
                data_group.write_to_storage(store_masks)
            return

        else:

            for iteration in xrange(self.augmentations[augmentation_num].total_iterations):

                for data_group in data_groups:

                    if augmentation_num == 0:
                        data_group.augmentation_cases[augmentation_num] = self.augmentations[augmentation_num].augmentation_dict[data_group.label].augment(data_group.base_case)
                    else:
                        data_group.augmentation_cases[augmentation_num] = self.augmentations[augmentation_num].augmentation_dict[data_group.label].augment(data_group.augmentation_cases[augmentation_num-1])

                    data_group.current_case = data_group.augmentation_cases[augmentation_num]
                    data_group.augmentation_num += 1

                self.recursive_augmentation(data_groups, augmentation_num+1)

                print 'FINISH RECURSION FOR AUGMENTATION NUM', augmentation_num+1

                for data_group in data_groups:
                    if augmentation_num == 0:
                        data_group.current_case = data_group.base_case
                    else:
                        data_group.current_case = data_group.augmentation_cases[augmentation_num - 1]


                for data_group in data_groups:        
                    self.augmentations[augmentation_num].augmentation_dict[data_group.label].iterate()

        for data_group in data_groups:
            data_group.augmentation_num -= 1

        return

class DataGroup(object):

    def __init__(self, label):

        self.label = label
        self.augmentations = []
        self.data = []
        self.cases = []

        # TODO: More distinctive naming for "base" and "current" cases.
        self.base_case = None
        self.base_casename = None
        self.base_affine = None

        self.augmentation_cases = []
        self.current_case = None

        self.augmentation_num = -1

        self.data_storage = None
        self.casename_storage = None
        self.affine_storage = None

        # TEMPORARY
        self.roimask_storage = None
        self.brainmask_storage = None
        self.roimask_outputpath = None
        self.brainmask_outputpath = None

        self.num_cases = 0

    def add_case(self, item, case_name):
        self.data.append(item)
        self.cases.append(case_name)
        self.num_cases = len(self.data)

    def append_augmentation(self, augmentation):
        self.augmentations.append(augmentation)
        self.augmentation_cases.append([])

    def get_augment_num_shape(self):

        output_num = len(self.data)
        output_shape = self.get_shape()

        # Get output size for list of augmentations.
        for augmentation in self.augmentations:

            # Error Catching
            if augmentation.total is None and augmentation.multiplier is None:
                continue

            # If multiplier goes over "total", use total
            if augmentation.total is None:
                output_num *= augmentation.multiplier
            elif ((num_cases * augmentation.multiplier) - num_cases) > augmentation.total:
                output_num += augmentation.total
            else:
                output_num *= augmentation.multiplier

            # Get output shape, if it changes
            if augmentation.output_shape is not None:
                output_shape = augmentation.output_shape

        return output_num, output_shape

    def get_shape(self):

        # TODO: Add support for non-nifti files.
        # Also this is not good. Perhaps specify shape in input?

        if self.data == []:
            return (0,)
        else:
            return nifti_2_numpy(self.data[0][0]).shape

    def get_modalities(self):
        if self.data == []:
            return 0
        else:
            return len(self.data[0])

    def augment(self, input_data):

        output_data = [input_data]

        for augmentatation in self.augmentations:

            output_data = augmentation.augment(input_data)

        return output_data

    def write_to_storage(self, store_masks=True):
        self.data_storage.append(self.current_case)
        self.casename_storage.append(np.array(self.base_casename)[np.newaxis][np.newaxis])
        self.affine_storage.append(self.base_affine[:][np.newaxis])

        if store_masks:
            self.roimask_storage.append(np.array(self.roimask_outputpath)[np.newaxis][np.newaxis])
            self.brainmask_storage.append(np.array(self.brainmask_outputpath)[np.newaxis][np.newaxis])

def create_hdf5_file(output_filepath, data_groups, data_collection, save_masks=False, store_masks=True):

    # Investigate hdf5 files.
    hdf5_file = tables.open_file(output_filepath, mode='w')

    # Investigate this line.
    # Compression levels = complevel. No compression = 0
    # Compression library = Method of compresion.
    filters = tables.Filters(complevel=5, complib='blosc')

    for data_group_label in data_groups:

        data_group = data_collection.data_groups[data_group_label]

        num_cases, output_shape = data_group.get_augment_num_shape()
        modalities = data_group.get_modalities()

        if num_cases == 0:
            raise FileNotFoundError('WARNING: No cases found. Cannot write to file.')

        # Input data has multiple 'channels' i.e. modalities.
        data_shape = tuple([0, modalities] + list(output_shape))

        data_group.data_storage = hdf5_file.create_earray(hdf5_file.root, data_group.label, tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=num_cases)

        # Naming convention is sketchy here, TODO, think about this.
        data_group.casename_storage = hdf5_file.create_earray(hdf5_file.root, '_'.join([data_group.label, 'casenames']), tables.StringAtom(256), shape=(0,1), filters=filters, expectedrows=num_cases)
        data_group.affine_storage = hdf5_file.create_earray(hdf5_file.root, '_'.join([data_group.label, 'affines']), tables.Float32Atom(), shape=(0,4,4), filters=filters, expectedrows=num_cases)

        if store_masks:
            data_group.roimask_storage = hdf5_file.create_earray(hdf5_file.root, '_'.join([data_group.label, 'roimask']), tables.StringAtom(256), shape=(0,1), filters=filters, expectedrows=num_cases)
            data_group.brainmask_storage = hdf5_file.create_earray(hdf5_file.root, '_'.join([data_group.label, 'brainmask']), tables.StringAtom(256), shape=(0,1), filters=filters, expectedrows=num_cases)

    return hdf5_file

def read_image_files(image_files, return_affine=False):

    image_list = []
    for i, image_file in enumerate(image_files):
        image_list.append(nifti_2_numpy(image_file))

    # This is a little clunky.
    if return_affine:
        # This assumes all images share an affine matrix.
        return np.stack([image for image in image_list]), nib.load(image_files[0]).affine
    else:
        return np.stack([image for image in image_list])

def save_masked_indice_list(input_data, brainmask_outputpath, roimask_outputpath=None, ground_truth_data=None, patch_shape=(16,16,16), mask_value=0):

    input_data = np.squeeze(input_data[:,0,...])
    ground_truth_data = np.squeeze(ground_truth_data)
    data_shape = input_data.shape

    nonzero_idx = input_data != mask_value
    nontumor_idx = ground_truth_data == mask_value

    brain_idx = np.asarray(np.where(np.logical_and(nonzero_idx, nontumor_idx))).T
    brain_idx = remove_invalid_idx(brain_idx, data_shape, patch_shape)

    tumor_idx = np.asarray(np.where(ground_truth_data > mask_value)).T
    tumor_idx = remove_invalid_idx(tumor_idx, data_shape, patch_shape)

    np.save(brainmask_outputpath, brain_idx)
    np.save(roimask_outputpath, tumor_idx)

    # return brain_idx, tumor_idx

def remove_invalid_idx(orignal_idx, shape, patch_size):

    idx1 = (orignal_idx + patch_size[0]/2)[:,0] < shape[0]
    idx2 = (orignal_idx + patch_size[1]/2)[:,1] < shape[1]
    idx3 = (orignal_idx + patch_size[2]/2)[:,2] < shape[2]
    idx4 = (orignal_idx - patch_size[0]/2)[:,0] >= 0
    idx5 = (orignal_idx - patch_size[1]/2)[:,1] >= 0
    idx6 = (orignal_idx - patch_size[2]/2)[:,2] >= 0

    valid = idx1 & idx2 & idx3 & idx4 & idx5 & idx6

    return orignal_idx[np.where(valid)[0],:]

def generate_idx(patient_dir,patch_size):
    os.chdir(patient_dir)
    seg = np.round(nib.load('seg_pp.nii.gz').get_data())
    FLAIR = np.round(nib.load('FLAIR_pp.nii.gz').get_data())
    
    nontumor_idx = np.asarray(np.where(seg==0)).T
    nonbackground_idx = np.asarray(np.nonzero(FLAIR)).T
    #normbrain = intersection of nontumor_idx and nonbackground_idx
    aset = set([tuple(x) for x in nontumor_idx])
    bset = set([tuple(x) for x in nonbackground_idx])
    normbrain_idx = np.array([x for x in aset & bset])
    valid_normbrain_idx = remove_invalid_idx(normbrain_idx,FLAIR.shape,patch_size)                     
    
    tumor_idx = np.asarray(np.where(seg>0)).T
    valid_tumor_idx = remove_invalid_idx(tumor_idx,FLAIR.shape,patch_size)
    
    np.save('normbrain_idx',valid_normbrain_idx)
    np.save('tumor_idx',valid_tumor_idx)

if __name__ == '__main__':
    pass