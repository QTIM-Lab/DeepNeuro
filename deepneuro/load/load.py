""" This library loads trained models, sample datasets, and other
        assets for use in DeepNeuro. By default, data is loaded into
        local directories in the [package_location]/deepneuro/load package 
        directory.
"""

import zipfile
import os

from six.moves.urllib.request import urlretrieve

current_dir = os.path.realpath(os.path.dirname(__file__))

# Perhaps one day replace this with config files distributed at the level of modules.
data_dict = {'skullstrip_mri': [os.path.join(current_dir, 'SkullStripping', 'Skullstrip_MRI_Model.h5'), 
                "https://www.dropbox.com/s/cucffmytzhp5byn/Skullstrip_MRI_Model.h5?dl=1"],

                'gbm_wholetumor_mri': [os.path.join(current_dir, 'Segment_GBM', 'Segment_GBM_Wholetumor_Model.h5'), 
                "https://www.dropbox.com/s/bnbdi1yogq2yye3/GBM_Wholetumor_Public.h5?dl=1"],

                'gbm_enhancingtumor_mri': [os.path.join(current_dir, 'Segment_GBM', 'Segment_GBM_Enhancing_Model.h5'),
                 "https://www.dropbox.com/s/hgsqi0vj7cfuk1g/GBM_Enhancing_Public.h5?dl=1"],

                'mets_enhancing': [os.path.join(current_dir, 'Segment_Mets', 'Segment_Mets_Model.h5'), 
                "https://www.dropbox.com/s/ea4xaput2lubuyw/Brain_Mets_Segmentation_Model.h5?dl=1"],

                'ischemic_stroke': [os.path.join(current_dir, 'Segment_Ischemic_Stroke', 'Ischemic_Stroke_Model.h5'),
                 'https://www.dropbox.com/s/4qpxvfac204xzhf/Ischemic_Stroke_Segmentation_Model.h5?dl=1'],

                'sample_gbm_nifti': [os.path.join(current_dir, 'Sample_Data', 'TCGA_GBM_NIFTI', 'TCGA_GBM_NIFTI.zip'),
                 'https://www.dropbox.com/s/bqclpqzwfsreolb/GBM_NIFTI.zip?dl=1'],
                 
                'sample_gbm_dicom': [os.path.join(current_dir, 'Sample_Data', 'TCGA_GBM_DICOM', 'TCGA_GBM_DICOM.zip'), 
                'https://www.dropbox.com/s/mbdq7m0vxutuwcs/GBM_DICOM.zip?dl=1']}


def load(dataset, output_datapath=None):

    """ This function loads trained models, sample datasets, and other
        assets for use in DeepNeuro. By default, data is loaded into
        local directories in the [package_location]/deepneuro/load package 
        directory.
    
    Parameters
    ----------
    dataset : str
        Key for dataset to be returned. E.g., "mets_enhancing"
    output_datapath: str
        Folder to output loaded data into. If None, will be placed in deepneuro directory.

    Returns
    -------
    str
        Output folder
    """

    # TODO: Loading progress indicator.

    if output_datapath is None:
        dataset_path = data_dict[dataset][0]
    else:
        dataset_path = os.path.join(output_datapath, os.path.basename(data_dict[dataset][0]))

    if not os.path.exists(os.path.dirname(dataset_path)):
        os.makedirs(os.path.dirname(dataset_path))

    if not os.path.exists(dataset_path):
        if True:
        # try:
            urlretrieve(data_dict[dataset][1], dataset_path)
            if dataset_path.endswith('.zip'):
                zip_ref = zipfile.ZipFile(dataset_path, 'r')
                zip_ref.extractall(os.path.dirname(dataset_path))
                zip_ref.close()
                os.remove(dataset_path)
            if dataset_path.endswith('.tar.gz'):
                raise NotImplementedError
        # except Exception:
            # if os.path.exists(dataset_path):
                # os.remove(dataset_path)
            # raise

    return dataset_path


if __name__ == '__main__':
    pass