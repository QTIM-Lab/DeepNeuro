## DeepNeuro

> A deep learning python package for medical imaging data.

## About

DeepNeuro is an open-source toolset of deep learning applications in medical imaging. We have several goals for this package:

* Provide easy-to-use command line tools for medical imaging using deep learning.
* Create Docker and Singularity containers for each tool and all out-of-package pre-processing steps, so they can each can be run without having install prerequisite libraries.
* Provide freely available deep learning models trained on a wealth of neuroimaging data.
* Provide training scripts and links to publically-available data to replicate the results of DeepNeuro's models.
* Provide implementations of popular models for medical imaging data, and pre-processed datasets for educational purposes.

To get started with DeepNeuro on your system, check out the [Installation](install.md) page.  
To run through some browser-based, GPU-enabled tutorials of DeepNeuro, see the [Tutorials](tutorials.md) page.  
To use DeepNeuro's pretrained modules, check out the [Modules](modules.md) page.  
To learn how to make your own deep learning pipelines in DeepNeuro, check out [Pipelines](pipelines.md).  
To read detailed documentation of all of DeepNeuro's functions, check out [Documentation](documentation.md).  
To read more about the architecture of DeepNeuro, and learn about how you can contribute to DeepNeuro, check out the [For Developers](developer.md) page.  

## Contact and Bug Reports

DeepNeuro was created by the Quantitative Tumor Imaging Lab at the Martinos Center (Harvard-MIT Program in Health, Sciences, and Technology / Massachusetts General Hospital). It is under active development.

If you run into an error, or have additional feature requests, please create an Issue on the [DeepNeuro Github page](https://github.com/QTIM-Lab/DeepNeuro/issues) describing your error or request. If you would like to contact the developers of DeepNeuro directly, please send a message to qtim.lab@gmail.com.

## Citation

If you use DeepNeuro in your published work, please cite:

Beers, A., Brown, J., Chang, K., Hoebel, K., Gerstner, E., Rosen, B., & Kalpathy-Cramer, J. (2018). DeepNeuro: an open-source deep learning toolbox for neuroimaging. arXiv preprint arXiv:1808.04589.

```@article{beers2018deepneuro, title={DeepNeuro: an open-source deep learning toolbox for neuroimaging}, author={Beers, Andrew and Brown, James and Chang, Ken and Hoebel, Katharina and Gerstner, Elizabeth and Rosen, Bruce and Kalpathy-Cramer, Jayashree}, journal={arXiv preprint arXiv:1808.04589}, year={2018}}
```

## Acknowledgements
The Center for Clinical Data Science at Massachusetts General Hospital and the Brigham and Woman's Hospital provided technical and hardware support for the development of DeepNeuro, including access to graphics processing units. The DeepNeuro project is also indebted to the following Github repository for the 3D UNet by user ellisdg, which formed the original kernel for much of its code in early stages. Long live open source deep learning :)