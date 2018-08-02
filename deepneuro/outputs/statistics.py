import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import label
from skimage.measure import regionprops

from deepneuro.utilities.conversion import convert_input_2_numpy


def calc_voxel_count(input_data, mask_value=0):
    
    input_data = convert_input_2_numpy(input_data)

    return input_data[input_data != mask_value].size


def calc_volume(image_numpy, pixdims, mask_value=0):
    return pixdims[0] * pixdims[1] * pixdims[2] * calc_voxel_count(image_numpy, mask_value)


def calc_RANO(input_label, affine=None, resolution_x=1, resolution_y=1, resolution_z=1, output_csv=None, image_data=None, output_image_filepath=None, display_image=False):

    """ Calculate RANO criteria. Assumes data is oriented [x, y, z]. TODO: Make dimension agnostic.
        Code modified from original written by Ken Chang.
        My editing of the code is not great; TODO: Refactor.
    """

    image_data, image_affine = convert_input_2_numpy(image_data, return_affine=True)
    input_label = convert_input_2_numpy(input_label)

    # if affine and image_affine is None:
        # pass
    # else:
        # resolution_x, resolution_y, resolution_z = 1, 1, 1  # Placeholder.

    rano_measures, rano_slices, rano_props = [], [], []

    connected_components = label(input_label, connectivity=2)
    component_labels = np.unique(connected_components)

    for lesion_idx in component_labels:

        lesion = connected_components.astype(int)
        lesion[connected_components != lesion_idx] = 0

        major_diameter, minor_diameter, rano_slice, region_props = [None] * 4

        volume_threshold = 2 * resolution_z
        if volume_threshold < 10:
            volume_threshold = 10

        for z_slice in range(lesion.shape[2]):

            lesion_slice = lesion[..., z_slice]

            if np.sum(lesion_slice) == 0:
                continue

            lesion_properties = regionprops(lesion_slice)
            current_major = lesion_properties[0].major_axis_length * resolution_x
            current_minor = lesion_properties[0].minor_axis_length * resolution_y

            if current_major < volume_threshold:
                continue
            if major_diameter is None:
                major_diameter, minor_diameter, rano_slice, region_props = current_major, current_minor, z_slice, lesion_properties
            elif current_major > major_diameter:
                major_diameter, minor_diameter, rano_slice, region_props = current_major, current_minor, z_slice, lesion_properties

        if major_diameter is not None:
            rano_measures += [major_diameter * minor_diameter]
            rano_slices += [rano_slice]
            rano_props += [region_props]

    if len(rano_measures) < 5:
        sum_rano = np.sum(rano_measures)
    else:
        sum_rano = np.sum(rano_measures.sort()[-5:])

    if output_csv is not None:
        if not os.path.exists(output_csv):
            pass

    if output_image_filepath is not None or display_image:

        for idx, z_slice in enumerate(rano_slices):

            lesion_props = rano_props[idx][0]

            if image_data is None:
                display_data = input_label[..., z_slice]
            else:
                display_data = image_data[..., z_slice]

            center_y, center_x = lesion_props.centroid
            major_angle = lesion_props.orientation

            minor_angle = major_angle + np.pi / 2

            half_major, half_minor = lesion_props.major_axis_length / 2, lesion_props.minor_axis_length / 2

            major_x_1 = center_x + np.cos(major_angle) * half_major
            major_y_1 = center_y - np.sin(major_angle) * half_major
            major_x_2 = center_x - np.cos(major_angle) * half_major
            major_y_2 = center_y + np.sin(major_angle) * half_major

            minor_x_1 = center_x + np.cos(minor_angle) * half_minor
            minor_y_1 = center_y - np.sin(minor_angle) * half_minor
            minor_x_2 = center_x - np.cos(minor_angle) * half_minor
            minor_y_2 = center_y + np.sin(minor_angle) * half_minor

            plt.imshow(display_data, interpolation='none', origin='lower', cmap='gray')
            plt.plot(center_x, center_y, 'ro') 
            plt.plot(major_x_1, major_y_1, 'ro') 
            plt.plot(major_x_2, major_y_2, 'ro')
            plt.plot(minor_x_1, minor_y_1, 'ro') 
            plt.plot(minor_x_2, minor_y_2, 'ro') 
            plt.show()

    return sum_rano


RANO = calc_RANO


def calculate_prediction_dice(self, label_volume_1, label_volume_2):

    im1 = np.asarray(label_volume_1).astype(np.bool)
    im2 = np.asarray(label_volume_2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return 0

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


if __name__ == '__main__':
    pass