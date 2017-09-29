import os
import numpy as np

from random import shuffle

def hdf5_data_generator(data_file, batch_size, data_labels, input_mapping_dict=None, augmentations=None):

    if isinstance(data_labels, basestring):
        data_labels = [data_labels]

    num_steps = getattr(data_file.root, data_labels[0]).shape[0]
    output_data_generator = data_generator(data_file, range(num_steps), data_labels=data_labels, batch_size=batch_size, augmentations=augmentations)

    return output_data_generator


def data_generator(data_file, index_list, data_labels, batch_size=1, augmentations=None):

    while True:
        data_lists = [[] for i in data_labels]
        shuffle(index_list)

        for index in index_list:

            add_data(data_lists, data_file, index, data_labels, augmentations)

            if len(data_lists[0]) == batch_size:

                yield tuple([np.asarray(data_list) for data_list in data_lists])
                data_lists = [[] for i in data_labels]


def add_data(data_lists, data_file, index, data_labels, augmentations=None):

    for data_idx, data_label in enumerate(data_labels):
        data = getattr(data_file.root, data_label)[index]
        data_lists[data_idx].append(data)