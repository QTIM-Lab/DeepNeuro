import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

from deepneuro.utilities.conversion import read_image_files


def create_mosaic(input_volume, output_filepath=None, label_volume=None, generate_outline=True, mask_value=0, step=1, dim=2, cols=8, label_buffer=5, rotate_90=3, flip=True):

    """This creates a mosaic of 2D images from a 3D Volume.
    
    Parameters
    ----------
    input_volume : TYPE
        Any neuroimaging file with a filetype supported by qtim_tools, or existing numpy array.
    output_filepath : None, optional
        Where to save your output, in a filetype supported by matplotlib (e.g. .png). If 
    label_volume : None, optional
        Whether to create your mosaic with an attached label filepath / numpy array. Will not perform volume transforms from header (yet)
    generate_outline : bool, optional
        If True, will generate outlines for label_volumes, instead of filled-in areas. Default is True.
    mask_value : int, optional
        Background value for label volumes. Default is 0.
    step : int, optional
        Will generate an image for every [step] slice. Default is 1.
    dim : int, optional
        Mosaic images will be sliced along this dimension. Default is 2, which often corresponds to axial.
    cols : int, optional
        How many columns in your output mosaic. Rows will be determined automatically. Default is 8.
    label_buffer : int, optional
        Images more than [label_buffer] slices away from a slice containing a label pixel will note be included. Default is 5.
    rotate_90 : int, optional
        If the output mosaic is incorrectly rotated, you may rotate clockwise [rotate_90] times. Default is 3.
    flip : bool, optional
        If the output is incorrectly flipped, you may set to True to flip the data. Default is True.
    
    No Longer Returned
    ------------------
    
    Returns
    -------
    output_array: N+1 or N-dimensional array
        The generated mosaic array.
    
    """

    image_numpy = read_image_files(input_volume)
    if step is None:
        step = 1

    if label_volume is not None:

        label_numpy = read_image_files(label_volume)

        if generate_outline:
            label_numpy = generate_label_outlines(label_numpy, dim, mask_value)

        # This is fun in a wacky way, but could probably be done more concisely and effeciently.
        mosaic_selections = []
        for i in range(label_numpy.shape[dim]):
            label_slice = np.squeeze(label_numpy[[slice(None) if k != dim else slice(i, i + 1) for k in range(3)]])
            if np.sum(label_slice) != 0:
                mosaic_selections += list(range(i - label_buffer, i + label_buffer))
        mosaic_selections = np.unique(mosaic_selections)
        mosaic_selections = mosaic_selections[mosaic_selections >= 0]
        mosaic_selections = mosaic_selections[mosaic_selections <= image_numpy.shape[dim]]
        mosaic_selections = mosaic_selections[::step]

        color_range_image = [np.min(image_numpy), np.max(image_numpy)]
        color_range_label = [np.min(label_numpy), np.max(label_numpy)]

        # One day, specify rotations by affine matrix.
        # Is test slice necessary? Operate directly on shape if possible.
        test_slice = np.rot90(np.squeeze(image_numpy[[slice(None) if k != dim else slice(0, 1) for k in range(3)]]), rotate_90)
        slice_width = test_slice.shape[1]
        slice_height = test_slice.shape[0]

        mosaic_image_numpy = np.zeros((int(slice_height * np.ceil(float(len(mosaic_selections)) / float(cols))), int(test_slice.shape[1] * cols)), dtype=float)
        mosaic_label_numpy = np.zeros_like(mosaic_image_numpy)
        
        row_index = 0
        col_index = 0

        for i in mosaic_selections:
            image_slice = np.rot90(np.squeeze(image_numpy[[slice(None) if k != dim else slice(i, i + 1) for k in range(3)]]), rotate_90)
            label_slice = np.rot90(np.squeeze(label_numpy[[slice(None) if k != dim else slice(i, i + 1) for k in range(3)]]), rotate_90)

            # Again, specify from affine matrix if possible.
            if flip:
                image_slice = np.fliplr(image_slice)
                label_slice = np.fliplr(label_slice)

            if image_slice.size > 0:
                mosaic_image_numpy[int(row_index):int(row_index + slice_height), int(col_index):int(col_index + slice_width)] = image_slice
                mosaic_label_numpy[int(row_index):int(row_index + slice_height), int(col_index):int(col_index + slice_width)] = label_slice

            if col_index == mosaic_image_numpy.shape[1] - slice_width:
                col_index = 0
                row_index += slice_height 
            else:
                col_index += slice_width

        mosaic_label_numpy = np.ma.masked_where(mosaic_label_numpy == 0, mosaic_label_numpy)

        if output_filepath is not None:
            plt.figure(figsize=(mosaic_image_numpy.shape[0] / 100, mosaic_image_numpy.shape[1] / 100), dpi=100, frameon=False)
            plt.margins(0, 0)
            plt.gca().set_axis_off()
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.imshow(mosaic_image_numpy, 'gray', vmin=color_range_image[0], vmax=color_range_image[1], interpolation='none')
            plt.imshow(mosaic_label_numpy, 'jet', vmin=color_range_label[0], vmax=color_range_label[1], interpolation='none')
            
            plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0.0, dpi=1000)
            plt.clf()
            plt.close()

        return mosaic_image_numpy

    else:

        color_range_image = [np.min(image_numpy), np.max(image_numpy)]

        test_slice = np.rot90(np.squeeze(image_numpy[[slice(None) if k != dim else slice(0, 1) for k in range(3)]]), rotate_90)
        slice_width = test_slice.shape[1]
        slice_height = test_slice.shape[0]

        mosaic_selections = np.arange(image_numpy.shape[dim])[::step]
        mosaic_image_numpy = np.zeros((int(slice_height * np.ceil(float(len(mosaic_selections)) / float(cols))), int(test_slice.shape[1] * cols)), dtype=float)

        row_index = 0
        col_index = 0

        for i in mosaic_selections:
            image_slice = np.squeeze(image_numpy[[slice(None) if k != dim else slice(i, i + 1) for k in range(3)]])

            image_slice = np.rot90(image_slice, rotate_90)
            
            if flip:
                image_slice = np.fliplr(image_slice)

            mosaic_image_numpy[int(row_index):int(row_index + slice_height), int(col_index):int(col_index + slice_width)] = image_slice

            if col_index == mosaic_image_numpy.shape[1] - slice_width:
                col_index = 0
                row_index += slice_height 
            else:
                col_index += slice_width

        if output_filepath is not None:
            plt.figure(figsize=(mosaic_image_numpy.shape[0] / 100, mosaic_image_numpy.shape[1] / 100), dpi=100, frameon=False)
            plt.margins(0, 0)
            plt.gca().set_axis_off()
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.imshow(mosaic_image_numpy, 'gray', vmin=color_range_image[0], vmax=color_range_image[1], interpolation='none')

            plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0.0, dpi=500) 
            plt.clf()
            plt.close()

        return mosaic_image_numpy


def generate_label_outlines(label_numpy, dim=2, mask_value=0):

    """ 
        Assumes labels are > 0 and integers.

        Parameters
        ----------

        input_volume: N-dimensional array
            The volume to be queried.
        mask_value: int or float
            Islands composed of "mask_value" will be ignored.
        return_split: bool
            Whether to a return a stacked output of equal-size binary arrays for each island,
            or to return one array with differently-labeled islands for each output.
        truncate: bool
            Whether or not to truncate the output. Irrelevant if return_split is False
        truncate_padding: int
            How many voxels of padding to leave when truncating.
        output_filepath: str
            If return_split is False, output will be saved to this file. If return_split
            is True, output will be save to this file with the suffix "_[#]" for island
            number

        Returns
        -------
        output_array: N+1 or N-dimensional array
            Output array(s) depending on return_split

    """
        
    edges_kernel = np.zeros((3, 3, 3), dtype=float)
    edges_kernel[1, 1, 1] = 4

    if dim != 2:
        edges_kernel[1, 1, 0] = -1
        edges_kernel[1, 1, 2] = -1

    if dim != 1:
        edges_kernel[1, 0, 1] = -1
        edges_kernel[1, 2, 1] = -1

    if dim != 0:
        edges_kernel[0, 1, 1] = -1
        edges_kernel[2, 1, 1] = -1
    
    outline_label_numpy = np.zeros_like(label_numpy, dtype=float)

    for label_number in np.unique(label_numpy):
        if label_number != mask_value:
            sublabel_numpy = np.copy(label_numpy)
            sublabel_numpy[sublabel_numpy != label_number] = 0
            edge_image = signal.convolve(sublabel_numpy, edges_kernel, mode='same').astype(int)
            edge_image[sublabel_numpy != label_number] = 0
            edge_image[edge_image != 0] = label_number
            outline_label_numpy += edge_image.astype(float)

    return outline_label_numpy


if __name__ == '__main__':
    pass