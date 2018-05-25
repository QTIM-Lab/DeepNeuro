"""DeepNeuro: A deep learning python package for neuroimaging data.

Created by the Quantitative Tumor Imaging Lab at the Martinos Center 
(Harvard-MIT Program in Health, Sciences, and Technology / Massachussets General Hospital).

"""

DOCLINES = __doc__.split("\n")

import sys

from setuptools import setup, find_packages
from codecs import open
from os import path

if sys.version_info[:2] < (2, 7):
    raise RuntimeError("Python version 2.7 or greater required.")

setup(
  name = 'deepneuro',
  # packages = ['qtim_tools'], # this must be the same as the name above
  version = '0.1.1',
  description = DOCLINES[0],
  packages = find_packages(),
  entry_points =  {
                  "console_scripts": ['segment_gbm = deepneuro.pipelines.Segment_GBM.cli:main',
                                      'skull_stripping = deepneuro.pipelines.Skull_Stripping.cli:main'], 
                  },
  author = 'Andrew Beers',
  author_email = 'abeers@mgh.harvard.edu',
  url = 'https://github.com/QTIM-Lab/DeepNeuro', # use the URL to the github repo
  download_url = 'https://github.com/QTIM-Lab/DeepNeuro/tarball/0.1.1',
  keywords = ['neuroimaging', 'neuroncology', 'neural networks', 'neuroscience', 'neurology', 'deep learning', 'fmri', 'pet', 'mri', 'dce', 'dsc', 'dti', 'machine learning', 'computer vision', 'learning', 'keras', 'theano', 'tensorflow', 'nfiti', 'nrrd', 'dicom'],
  install_requires=['keras', 'pydicom', 'pynrrd', 'nibabel', 'numpy', 'scipy', 'scikit-image==0.12.3'],
  classifiers = [],
)