import numpy as np
import matplotlib.pyplot as plt

from deepneuro.utilities.conversion import save_data
from deepneuro.utilities.util import replace_suffix


def check_data(output_data=None, data_collection=None, batch_size=4, merge_batch=True, show_output=True, output_filepath=None, viz_rows=None, viz_mode_2d=None, viz_mode_3d='2d_center', color_range=None, output_groups=None, combine_outputs=False, rgb_output=True):

    if data_collection is not None:
        generator = data_collection.data_generator(perpetual=True, verbose=False, batch_size=batch_size)
        output_data = next(generator)
    
    if type(output_data) is not dict:
        output_data = {'output_data': output_data}

    if color_range is None:
        color_range = {label: [np.min(data), np.max(data)] for label, data in list(output_data.items())}

    if output_groups is not None:
        output_data = {label: data for label, data in list(output_data.items()) if label in output_groups}

    output_images = {}

    if viz_rows is None:
        viz_rows = int(np.ceil(np.sqrt(batch_size)))

    viz_rows = min(viz_rows, batch_size)
    viz_columns = int(np.ceil(batch_size / float(viz_rows)))

    for label, data in list(output_data.items()):

        if data.ndim == 5:
            output_images, color_range = display_3d_data(data, color_range, viz_mode_3d, label, output_images, viz_rows, viz_columns)

        elif data.ndim == 4:
            if data.shape[-1] == 2:
                for i in range(data.shape[-1]):
                    output_images[label + '_' + str(i)] = merge_data(data[..., i][..., np.newaxis], [viz_rows, viz_columns], 1)
                    color_range[label + '_' + str(i)] = color_range[label]
            if data.shape[-1] not in [1, 3]:
                output_images[label + '_RGB'] = merge_data(data[..., 0:3], [viz_rows, viz_columns], 3)
                color_range[label + '_RGB'] = color_range[label]
                for i in range(3, data.shape[-1]):
                    output_images[label + '_' + str(i)] = merge_data(data[..., i][..., np.newaxis], [viz_rows, viz_columns], 1)
                    color_range[label + '_' + str(i)] = color_range[label]
            else:
                output_images[label] = merge_data(data, [viz_rows, viz_columns], data.shape[-1])

    if show_output:

        plots = len(list(output_images.keys()))
        plot_rows = int(np.ceil(np.sqrt(plots)))
        plot_columns = int(np.ceil(plots / float(plot_rows)))
        fig, axarr = plt.subplots(plot_rows, plot_columns)

        # matplotlib is so annoying
        if plot_rows == 1 and plot_columns == 1:
            axarr = np.array([axarr]).reshape(1, 1)
        elif plot_rows == 1 or plot_columns == 1:
            axarr = axarr.reshape(plot_rows, plot_columns)

        for plot_idx, (label, data) in enumerate(output_images.items()):

            image_column = plot_idx % plot_columns
            image_row = plot_idx // plot_columns

            if data.shape[-1] == 3:

                # Weird matplotlib bug/feature:
                if np.min(data) < 0:
                    data = (data - np.min(data)) / (np.max(data) - np.min(data))

                plt_image = axarr[image_row, image_column].imshow(np.squeeze(data), cmap=plt.get_cmap('hot'), vmin=color_range[label][0], vmax=color_range[label][1], interpolation='none')

                fig.colorbar(plt_image, ax=axarr[image_row, image_column])

            elif data.shape[-1] == 1:
                plt_image = axarr[image_row, image_column].imshow(np.squeeze(data), cmap='gray', vmin=color_range[label][0], vmax=color_range[label][1], interpolation='none')

                fig.colorbar(plt_image, ax=axarr[image_row, image_column], cmap='gray')

            axarr[image_row, image_column].set_title(label)

        plt.show()

    output_filepaths = {}
    for label, data in list(output_images.items()):
        output_images[label] = image_preprocess(data)
        if output_filepath is not None:
            output_filepaths[label] = save_data(output_images[label], replace_suffix(output_filepath, '', '_' + label))

    return output_filepaths, output_images


def image_preprocess(input_data):

    if input_data.ndim == 3:
        if np.min(input_data) == np.max(input_data):
            input_data[:] = 0
            return input_data
        else:
            input_data = ((255) * (input_data - np.min(input_data))) / (np.max(input_data) - np.min(input_data))
            return input_data


def combine_outputs(input_data_list):

    """ Merges images rows
    """

    # height_width = [0, 0]

    for data in input_data_list:

        pass

    raise NotImplementedError


def display_3d_data(input_data, color_range, viz_mode_3d='2d_center', label=None, input_dict=None, viz_rows=2, viz_columns=2):

    if input_dict is None:
        input_dict = {}

    if label is None:
        label = 'input_data'

    for i in range(input_data.shape[-1]):

        if viz_mode_3d == '2d_center':

            input_data_slice = input_data[..., int(input_data.shape[-1] / 2), i][..., np.newaxis]
            input_data_slice = merge_data(input_data_slice, [viz_rows, viz_columns], 1)

        else:

            raise NotImplementedError

        if input_dict is not None:

            if input_data.shape[-1] == 1:
                input_dict[label] = input_data_slice
            else:
                input_dict[label + '_' + str(i)] = input_data_slice
                color_range[label + '_' + str(i)] = color_range[label]

    return input_dict, color_range


def merge_data(images, size, channels=3):

    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], channels))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img