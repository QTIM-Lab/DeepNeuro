""" Testing functions for 
"""


def test_glioblastoma_module(testing_directory="/home/DeepNeuro/tmp", gpu_num='0'):

    import numpy as np
    import os
    from shutil import rmtree

    FLAIR, T1POST, T1PRE = np.random.normal(loc=1000, scale=200, size=(240, 240, 40)), \
                            np.random.normal(loc=1500, scale=200, size=(240, 240, 180)), \
                            np.random.normal(loc=1300, scale=200, size=(120, 120, 60))

    from deepneuro.utilities.conversion import save_data

    try:
        os.mkdir(testing_directory)
        FLAIR_file = save_data(FLAIR, os.path.join(testing_directory, 'FLAIR.nii.gz'))
        T1PRE_file = save_data(T1PRE, os.path.join(testing_directory, 'T1PRE.nii.gz'))
        T1POST_file = save_data(T1POST, os.path.join(testing_directory, 'T1POST.nii.gz'))

        from deepneuro.pipelines.Segment_GBM.predict import predict_GBM

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
        predict_GBM(testing_directory, 
                T1POST=T1POST_file, 
                FLAIR=FLAIR_file, 
                T1PRE=T1PRE_file)

        rmtree(testing_directory)

    except:
        rmtree(testing_directory)
        raise

    return


if __name__ == '__main__':

    pass