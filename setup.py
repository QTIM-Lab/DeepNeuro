"""DeepNeuro: A deep learning python package for neuroimaging data.

Created by the Quantitative Tumor Imaging Lab at the Martinos Center 
(Harvard-MIT Program in Health, Sciences, and Technology / Massachussets General Hospital).

"""

DOCLINES = __doc__.split("\n")

import sys

from setuptools import setup, find_packages
from codecs import open
from os import path
import os

os.environ["MPLCONFIGDIR"] = "."

if sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version 3.5 or greater required.")

setup(
  name='deepneuro',
  version='0.2.3',
  description=DOCLINES[0],
  packages=find_packages(),
  entry_points= {
                  "console_scripts": ['segment_gbm = deepneuro.pipelines.Segment_GBM.cli:main',
                                      'skull_stripping = deepneuro.pipelines.Skull_Stripping.cli:main',
                                      'segment_mets = deepneuro.pipelines.Segment_Brain_Mets.cli:main',
                                      'segment_ischemic_stroke = deepneuro.pipelines.Ischemic_Stroke.cli:main'], 
                  },
  author='Andrew Beers',
  author_email='abeers@mgh.harvard.edu',
  url='https://github.com/QTIM-Lab/DeepNeuro',  # use the URL to the github repo
  download_url='https://github.com/QTIM-Lab/DeepNeuro/tarball/0.2.3',
  keywords=['neuroimaging', 'neuroncology', 'neural networks', 'neuroscience', 'neurology', 'deep learning', 'fmri', 'pet', 'mri', 'dce', 'dsc', 'dti', 'machine learning', 'computer vision', 'learning', 'keras', 'theano', 'tensorflow', 'nifti', 'nrrd', 'dicom'],
  install_requires=['tables', 'pydicom', 'pynrrd', 'nibabel', 'pyyaml', 'six', 'imageio', 'matplotlib', 'pydot', 'scipy', 'numpy', 'scikit-image', 'imageio', 'tqdm'],
  classifiers=[],
)
