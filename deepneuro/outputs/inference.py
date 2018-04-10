
import sys

from deepneuro.outputs.output import Output
from deepneuro.utilities.util import add_parameter

import numpy as np


class ModelInference(Output):

    def load(self, kwargs):

        """ Parameters
            ----------
            depth : int, optional
                Specified the layers deep the proposed U-Net should go.
                Layer depth is symmetric on both upsampling and downsampling
                arms.
            max_filter: int, optional
                Specifies the number of filters at the bottom level of the U-Net.

        """

        # Evaluation Params
        add_parameter(self, kwargs, 'ground_truth', None)

        # Saving Params
        add_parameter(self, kwargs, 'postprocessor_string', '_pseudoprobability')

        # Model Parameters
        add_parameter(self, kwargs, 'channels_first', False)
        add_parameter(self, kwargs, 'input_channels', None)

        if 'channels_dim' in kwargs:
            self.channels_dim = kwargs.get('channels_dim')
        elif self.channels_first:
            self.channels_dim = 1
        else:
            self.channels_dim = -1

    def process_case(self, input_data):

        # A little bit strange to access casename this way. Maybe make it an optional
        # return of the generator.

        # Note that input_modalities as the first input is hard-coded here. Very fragile.

        # If an image is being repatched, its output shape is not certain. We attempt to infer it from
        # the input data. This is wonky. Move this to PatchInference, maybe.

        if self.channels_first:
            input_data = np.swapaxes(input_data[0], 1, -1)
        else:
            # Temporary code. In the future, make sure code works with multiple and specific inputs.
            input_data = input_data[0]

        if self.input_channels is not None:
            input_data = np.take(input_data, self.input_channels, self.channels_dim)

        self.output_shape = [1] + list(self.model.model.layers[-1].output_shape)[1:]  # Weird
        for i in xrange(len(self.patch_dimensions)):
            self.output_shape[self.output_patch_dimensions[i]] = input_data.shape[self.patch_dimensions[i]]

        output_data = self.predict(input_data)

        # Will fail for time-data.
        if self.channels_first:
            output_data = np.swapaxes(output_data, 1, -1)

        self.return_objects.append(output_data)

    def predict(self, input_data, model, batch_size):

        # Vanilla prediction case is obivously not fleshed out.
        prediction = model.predict(input_data)

        return prediction


class ModelPatchesInference(ModelInference):

    def load(self, kwargs):

        """ Parameters
            ----------
            depth : int, optional
                Specified the layers deep the proposed U-Net should go.
                Layer depth is symmetric on both upsampling and downsampling
                arms.
            max_filter: int, optional
                Specifies the number of filters at the bottom level of the U-Net.

        """

        super(ModelPatchesInference, self).load(kwargs)

        if 'patch_overlaps' in kwargs:
            self.patch_overlaps = kwargs.get('patch_overlaps')
        else:
            self.patch_overlaps = 1

        if 'patch_dimensions' in kwargs:
            self.patch_dimensions = kwargs.get('patch_dimensions')
        else:
            # TODO: Set better defaults.
            if self.channels_first:
                self.patch_dimensions = [-3, -2, -1]
            else:
                self.patch_dimensions = [-4, -3, -2]

        # A little tricky to not refer to previous paramter as input_patch_dimensions
        if 'output_patch_dimensions' in kwargs:
            self.output_patch_dimensions = kwargs.get('output_patch_dimensions')
        else:
            self.output_patch_dimensions = self.patch_dimensions

        if 'output_patch_shape' in kwargs:
            self.output_patch_shape = kwargs.get('output_patch_shape')
        else:
            self.output_patch_shape = None

        if 'pad_borders' in kwargs:
            self.pad_borders = kwargs.get('pad_borders')
        else:
            self.pad_borders = True

        if 'check_empty_patch' in kwargs:
            self.check_empty_patch = kwargs.get('check_empty_patch')
        else:
            self.check_empty_patch = True

    def generate(self):

        # Determine patch shape. Currently only extends to spatial patching.
        # This leading dims business has got to have a better solution..
        self.input_patch_shape = self.model.model.layers[0].input_shape
        if self.output_patch_shape is None:
            self.output_patch_shape = self.model.model.layers[-1].output_shape

        return super(ModelPatchesInference, self).generate()

    def predict(self, input_data):

        repetition_offsets = [np.linspace(0, self.input_patch_shape[axis] - 1, self.patch_overlaps, dtype=int) for axis in self.patch_dimensions]

        if self.pad_borders:
            # TODO -- Clean up this border-padding code and make it more readable.
            input_pad_dimensions = [(0, 0)] * input_data.ndim
            repatched_shape = self.output_shape
            new_input_shape = list(input_data.shape)
            for idx, dim in enumerate(self.patch_dimensions):
                # Might not work for odd-shaped patches; check.
                input_pad_dimensions[dim] = (int(self.input_patch_shape[dim] / 2), int(self.input_patch_shape[dim] / 2))
                new_input_shape[dim] += self.input_patch_shape[dim]
            for idx, dim in enumerate(self.output_patch_dimensions):
                repatched_shape[dim] += self.input_patch_shape[dim]

            padded_input_data = np.zeros(new_input_shape)
            input_slice = [slice(None)] + [slice(self.input_patch_shape[dim] / 2, -self.input_patch_shape[dim] / 2, None) for dim in self.patch_dimensions]
            padded_input_data[input_slice] = input_data
            input_data = padded_input_data

        repatched_image = np.zeros(repatched_shape)

        corner_data_dims = [input_data.shape[axis] for axis in self.patch_dimensions]
        corner_patch_dims = [self.output_patch_shape[axis] for axis in self.patch_dimensions]

        all_corners = np.indices(corner_data_dims)

        # There must be a better way to round up to an integer..
        possible_corners_slice = [slice(None)] + [slice(self.input_patch_shape[dim] / 2, -self.input_patch_shape[dim] / 2, None) for dim in self.patch_dimensions]
        all_corners = all_corners[possible_corners_slice]

        for rep_idx in xrange(self.patch_overlaps):

            if self.verbose:
                print 'Patch prediction repetition', rep_idx
                sys.stdout.flush()

            corners_grid_shape = [slice(None)]
            for dim in xrange(all_corners.ndim - 1):
                corners_grid_shape += [slice(repetition_offsets[dim][rep_idx], corner_data_dims[dim], corner_patch_dims[dim])]

            corners_list = all_corners[corners_grid_shape]
            corners_list = np.reshape(corners_list, (corners_list.shape[0], -1)).T

            if self.check_empty_patch:
                corners_list = self.remove_empty_patches(input_data, corners_list)

            for corner_list_idx in xrange(0, corners_list.shape[0], self.batch_size):

                corner_batch = corners_list[corner_list_idx:corner_list_idx + self.batch_size]
                input_patches = self.grab_patch(input_data, corner_batch)

                prediction = self.model.model.predict(input_patches)
                
                self.insert_patch(repatched_image, prediction, corner_batch)

            if rep_idx == 0:
                output_data = np.copy(repatched_image)
            else:
                output_data = output_data + (1.0 / (rep_idx)) * (repatched_image - output_data)  # Running Average

        if self.pad_borders:

            output_slice = [slice(None)] * output_data.ndim  # Weird
            for idx, dim in enumerate(self.output_patch_dimensions):
                # Might not work for odd-shaped patches; check.
                output_slice[dim] = slice(self.input_patch_shape[dim] / 2, -self.input_patch_shape[dim] / 2, 1)
            output_data = output_data[output_slice]

        return output_data

    def pad_data(self, data, pad_dimensions):

        # Maybe more effecient than np.pad? Created for testing a different purpose.

        for idx, width in enumerate(pad_dimensions):
            pad_block_1, pad_block_2 = list(data.shape), list(data.shape)
            pad_block_1[idx] = width[0]
            pad_block_2[idx] = width[1]
            data = np.concatenate((np.zeros(pad_block_1), data, np.zeros(pad_block_2)), axis=idx)

        return data

    def remove_empty_patches(self, input_data, corners_list):

        corner_selections = []

        for corner_idx, corner in enumerate(corners_list):
            output_slice = [slice(None)] * input_data.ndim  # Weird
            for idx, dim in enumerate(self.patch_dimensions):
                output_slice[dim] = slice(corner[idx] - self.input_patch_shape[dim] / 2, corner[idx] + self.input_patch_shape[dim] / 2, 1)

            corner_selections += [np.any(input_data[output_slice])]

        return corners_list[corner_selections]

    def grab_patch(self, input_data, corner_list):

        """ Given a corner coordinate, a patch_shape, and some input_data, returns a patch or array of patches.
        """

        output_patches_shape = (corner_list.shape[0], ) + self.input_patch_shape[1:]
        output_patches = np.zeros((output_patches_shape))

        for corner_idx, corner in enumerate(corner_list):
            output_slice = [slice(None)] * input_data.ndim  # Weird
            for idx, dim in enumerate(self.patch_dimensions):
                output_slice[dim] = slice(corner[idx] - self.input_patch_shape[dim] / 2, corner[idx] + self.input_patch_shape[dim] / 2, 1)

            output_patches[corner_idx, ...] = input_data[output_slice]

        return output_patches

    def insert_patch(self, input_data, patches, corner_list):

        # Some ineffeciencies in the function. TODO: come back and rewrite.

        for corner_idx, corner in enumerate(corner_list):
            insert_slice = [slice(None)] * input_data.ndim  # Weird
            for idx, dim in enumerate(self.output_patch_dimensions):
                # Might not work for odd-shaped patches; check.
                insert_slice[dim] = slice(corner[idx] - self.output_patch_shape[dim] / 2, corner[idx] + self.output_patch_shape[dim] / 2, 1)

            insert_patch = patches[corner_idx, ...]
            if not np.array_equal(np.take(self.output_patch_shape, self.output_patch_dimensions), np.take(self.input_patch_shape, self.patch_dimensions)):  # Necessary if statement?
                patch_slice = [slice(None)] * insert_patch.ndim  # Weird
                for idx, dim in enumerate(self.output_patch_dimensions):
                    # Might not work for odd-shaped patches; check.
                    patch_slice[dim] = slice((self.input_patch_shape[dim] - self.output_patch_shape[dim]) / 2, -(self.input_patch_shape[dim] - self.output_patch_shape[dim]) / 2, 1)

                insert_patch = insert_patch[patch_slice]

            input_data[insert_slice] = insert_patch

        return input_data