""" A function library for tools to visualize data. At present, only hosts
    the check_data function, which allows one to inspect DataCollections and
    other freeform data.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from collections import OrderedDict

from deepneuro.utilities.conversion import save_data, read_image_files
from deepneuro.utilities.util import replace_suffix, nifti_splitext


def check_data(output_data=None, data_collection=None, batch_size=4, merge_batch=True, show_output=True, output_filepath=None, viz_rows=None, viz_mode_2d=None, viz_mode_3d='2d_slice', color_range=None, output_groups=None, combine_outputs=False, rgb_output=True, colorbar=True, subplot_rows=None, title=None, subplot_titles=None, output_directory=None, case_name=None, **kwargs):

    if data_collection is not None:
        if batch_size > data_collection.total_cases * data_collection.multiplier:
            batch_size = data_collection.total_cases * data_collection.multiplier

        generator = data_collection.data_generator(perpetual=True, verbose=False, batch_size=batch_size)
        case_name = data_collection.current_case
        output_data = next(generator)

    if type(output_data) is not dict:
        output_data = {'output_data': output_data}

    # Very bad, reformat.
    output_data = {label: data for label, data in list(output_data.items()) if '_affine' not in label and '_augmentation_string' not in label and 'casename' not in label}

    for label, data in list(output_data.items()):
        if type(data) is str:
            output_data[label] = read_image_files(data)[np.newaxis, ...]

    if color_range is None:
        color_range = {label: [np.min(data), np.max(data)] for label, data in list(output_data.items())}

    if output_groups is not None:
        output_data = {label: data for label, data in list(output_data.items()) if label in output_groups}

    if not show_output:
        subplot_titles = None

    output_images = OrderedDict()

    if data_collection is None:
        reference_key = list(output_data)[0]
        batch_size = output_data[reference_key].shape[0]

    if viz_rows is None:
        viz_rows = int(np.ceil(np.sqrt(batch_size)))
    else:
        viz_rows = min(viz_rows, batch_size)

    viz_columns = int(np.ceil(batch_size / float(viz_rows)))

    for label, data in list(output_data.items()):

        if '_affine' in label or '_augmentation_string' in label or 'casename' in label:
            continue

        if data.ndim == 5:
            output_images, color_range = display_3d_data(data, color_range, viz_mode_3d, label, output_images, viz_rows, viz_columns, subplot_titles=subplot_titles, **kwargs)

        elif data.ndim == 4:

            if data.shape[-1] == 2:
                for i in range(data.shape[-1]):
                    
                    if subplot_titles is None:
                        subplot_title = label + '_' + str(i)
                    else:
                        subplot_title = subplot_titles[label][i]

                    output_images[subplot_title] = merge_data(data[..., i][..., np.newaxis], [viz_rows, viz_columns], 1)
                    color_range[subplot_title] = color_range[label]

            if data.shape[-1] not in [1, 3]:

                output_images[label + '_RGB'] = merge_data(data[..., 0:3], [viz_rows, viz_columns], 3)
                color_range[label + '_RGB'] = color_range[label]
                for i in range(3, data.shape[-1]):
                    output_images[label + '_' + str(i)] = merge_data(data[..., i][..., np.newaxis], [viz_rows, viz_columns], 1)
                    color_range[label + '_' + str(i)] = color_range[label]
            else:

                output_images[label] = merge_data(data, [viz_rows, viz_columns], data.shape[-1])

        elif data.ndim == 3:

            output_images[label] = merge_data(data, [viz_rows, viz_columns], data.shape[-1])

        # elif data.ndim == 2:

        #     output_images[label] = display_1d_data(data, [viz_rows, viz_columns])

    if show_output:

        plots = len(list(output_images.keys()))
        if subplot_rows is None:
            subplot_rows = int(np.ceil(np.sqrt(plots)))
        plot_columns = int(np.ceil(plots / float(subplot_rows)))
        fig, axarr = plt.subplots(plot_columns, subplot_rows)

        # matplotlib is so annoying
        if subplot_rows == 1 and plot_columns == 1:
            axarr = np.array([axarr]).reshape(1, 1)
        elif subplot_rows == 1 or plot_columns == 1:
            axarr = axarr.reshape(plot_columns, subplot_rows)

        for plot_idx, (label, data) in enumerate(output_images.items()):

            image_row = plot_idx % plot_columns
            image_column = plot_idx // plot_columns

            if data.shape[-1] == 3:

                # Weird matplotlib bug/feature:
                if np.min(data) < 0:
                    data = (data - np.min(data)) / (np.max(data) - np.min(data))

                plt_image = axarr[image_row, image_column].imshow(np.squeeze(data), cmap=plt.get_cmap('hot'), vmin=color_range[label][0], vmax=color_range[label][1], interpolation='none')

                if colorbar:
                    fig.colorbar(plt_image, ax=axarr[image_row, image_column])

            elif data.shape[-1] == 1:
                plt_image = axarr[image_row, image_column].imshow(np.squeeze(data), cmap='gray', vmin=color_range[label][0], vmax=color_range[label][1], interpolation='none')

                if colorbar:
                    fig.colorbar(plt_image, ax=axarr[image_row, image_column], cmap='gray')

            axarr[image_row, image_column].set_title(label)

        for plot_idx in range(len(output_images), subplot_rows * plot_columns):
            image_row = plot_idx % plot_columns
            image_column = plot_idx // plot_columns
            fig.delaxes(axarr[image_row, image_column])

        if title is not None:
            fig.suptitle(title, fontsize=28)

        plt.show()

    output_filepaths = {}
    if (output_directory is not None or output_filepath is not None):
        for label, data in list(output_images.items()):
            output_images[label] = image_preprocess(data)
            if output_filepath is not None:
                output_filepaths[label] = save_data(output_images[label], replace_suffix(output_filepath, '', '_' + label))
                continue
            if output_directory is not None and case_name is not None:
                output_filepaths[label] = save_data(output_images[label], os.path.join(output_directory, nifti_splitext(replace_suffix(os.path.basename(case_name), '', '_' + label))[0]) + '.png')

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

    raise NotImplementedError


def display_1d_data(input_data, viz_rows=2, viz_columns=2):

    class_viz_rows = int(np.ceil(np.sqrt(input_data.shape[-1])))

    output_data = np.zeros((class_viz_rows * viz_rows, class_viz_rows * viz_columns))

    raise NotImplementedError

    return output_data


def display_2d_data(input_data):

    raise NotImplementedError

    return


def display_3d_data(input_data, color_range, viz_mode_3d='2d_slice', label=None, input_dict=None, viz_rows=2, viz_columns=2, slice_index=None, mosaic_rows=4, mosaic_columns=4, subplot_titles=None, **kwargs):

    if input_dict is None:
        input_dict = {}

    if label is None:
        label = 'input_data'

    for i in range(input_data.shape[-1]):

        if viz_mode_3d == '2d_slice':

            if slice_index is None:
                input_data_slice = input_data[..., int(input_data.shape[-2] / 2), i][..., np.newaxis]
            else:
                input_data_slice = input_data[..., slice_index, i][..., np.newaxis]
            input_data_slice = merge_data(input_data_slice, [viz_rows, viz_columns], 1)

        elif viz_mode_3d == 'mosaic':

            input_data_slice = np.zeros((input_data.shape[0], mosaic_rows * input_data.shape[1], mosaic_columns * input_data.shape[2], 1), dtype=input_data.dtype)

            image_idx = 0
            slice_gap = input_data.shape[3] // (mosaic_rows * mosaic_columns)
            for m_row in range(mosaic_rows):
                for m_col in range(mosaic_columns):
                    input_data_slice[:, m_row * input_data.shape[1]: (m_row + 1) * input_data.shape[1], m_col * input_data.shape[2]: (m_col + 1) * input_data.shape[2]] = input_data[:, ..., slice_gap * image_idx, i][..., np.newaxis]
                    image_idx += 1

            input_data_slice = merge_data(input_data_slice, [viz_rows, viz_columns], 1)

        else:

            raise NotImplementedError

        if input_dict is not None:

            if subplot_titles is None:
                if input_data.shape[-1] == 1:
                    subplot_title = label
                else:
                    subplot_title = label + '_' + str(i)
            else:
                subplot_title = subplot_titles[label][i]

            input_dict[subplot_title] = input_data_slice
            color_range[subplot_title] = [np.min(input_data[..., i]), np.max(input_data[..., i])]

    return input_dict, color_range


def merge_data(images, size, channels=3):

    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], channels))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img