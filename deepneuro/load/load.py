
import os
import glob
import urllib

def load(dataset):

    if dataset == 'Segment_GBM_downsampled_edema':
        dataset_path = os.path.join(os.path.realpath(__file__), 'Segment_GBM', 'downsampled_edema.h5')
        if not os.path.exists(dataset_path):
            urllib.urlretrieve("https://www.dropbox.com/s/h2pj9hzruz3hq9a/wholetumor_downsampled_BRATS_submission.h5?dl=1", dataset_path)
        return dataset_path

if __name__ == '__main__':
    pass