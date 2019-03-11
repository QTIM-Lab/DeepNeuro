import numpy as np

from deepneuro.outputs.inference import ModelInference
from deepneuro.utilities.util import add_parameter, docker_print


class PatchesInference(ModelInference):

    """
    """

    def load(self, kwargs):

        """ Parameters
            ----------
            patch_overlaps: int, optional
                The amount of times a grid of patches is predicted over an entire
                output volume. Subsequent grids are offset from the original grid
                by patch_size / patch_overlaps, and the final output is the result
                of averaging over each grid for each voxel. Default is 1.
            input_patch_shape: tuple, optional
                The input dimensions of the predicted patches, not including batch
                size. If None, DeepNeuro will attempt to extract this value from the
                given model. Default is None.
            output_patch_shape: tuple, optional
                The output dimensions of the predicted patches, not including batch
                size. If smaller than the input patch size in any dimension, patches
                will be cropped symmetrically by the difference in size to meet this
                shape. Default is None.
            check_empty_patch: bool, optional
                Do not predict patches if they only contain zeros. Default is True.
            pad_borders: bool, optional
                Pads input borders by patch_size / 2 with zeros. This allows patches
                at the boundary of an image to be successfully predicted, albeit with
                zero infill values. Default is True.
            patch_dimensions: tuple or list, optional
            output_patch_dimensions: tuple or list, optional

        """

        super(PatchesInference, self).load(kwargs)

        # Patching Parameters
        add_parameter(self, kwargs, 'patch_overlaps', 1)
        add_parameter(self, kwargs, 'input_patch_shape', None)
        add_parameter(self, kwargs, 'output_patch_shape', None)
        add_parameter(self, kwargs, 'check_empty_patch', True)
        add_parameter(self, kwargs, 'pad_borders', True)
        add_parameter(self, kwargs, 'keep_channels', None)

        add_parameter(self, kwargs, 'patch_dimensions', None)
        add_parameter(self, kwargs, 'output_patch_dimensions', self.patch_dimensions)

        self.batch_size = 1

    def process_case(self, input_data, model=None):
        
        """Summary
        
        Parameters
        ----------
        input_data : TYPE
            Description
        model : None, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        
        input_data = input_data[self.lead_key]
        
        if model is not None:
            self.model = model

        if self.channels_first:
            input_data = np.swapaxes(input_data, 1, -1)

        if self.input_channels is not None:
            input_data = np.take(input_data, self.input_channels, self.channels_dim)

        # Determine patch shape. Currently only extends to spatial patching.
        # This leading dims business has got to have a better solution..
        if self.input_patch_shape is None:
            self.input_patch_shape = self.model.model_input_shape
        if self.output_patch_shape is None:
            self.output_patch_shape = self.model.model_output_shape

        self.input_dim = len(self.input_patch_shape) - 2

        if self.patch_dimensions is None:
            if self.channels_first:
                self.patch_dimensions = [-1 * self.input_dim + x for x in range(self.input_dim)]
            else:
                self.patch_dimensions = [-1 * self.input_dim + x - 1 for x in range(self.input_dim)]

            if self.output_patch_dimensions is None:
                self.output_patch_dimensions = self.patch_dimensions

        self.output_shape = [1] + list(self.model.model_output_shape)[1:]  # Weird
        for i in range(len(self.patch_dimensions)):
            self.output_shape[self.output_patch_dimensions[i]] = input_data.shape[self.patch_dimensions[i]]

        output_data = self.predict(input_data)

        if self.output_channels is not None:
            output_data = np.take(output_data, self.output_channels, self.channels_dim)

        # Will fail for time-data.
        if self.channels_first:
            output_data = np.swapaxes(output_data, 1, -1)

        self.return_objects.append(output_data)

        return output_data

    def predict(self, input_data):

        repetition_offsets = [np.linspace(0, self.input_patch_shape[axis] - 1, self.patch_overlaps + 1, dtype=int)[:-1] for axis in self.patch_dimensions]

        if self.pad_borders:
            # TODO -- Clean up this border-padding code and make it more readable.
            input_pad_dimensions = [(0, 0)] * input_data.ndim
            repatched_shape = self.output_shape
            new_input_shape = list(input_data.shape)
            for idx, dim in enumerate(self.patch_dimensions):
                # Might not work for odd-shaped patches; check.
                input_pad_dimensions[dim] = (int(self.input_patch_shape[dim] // 2), int(self.input_patch_shape[dim] // 2))
                new_input_shape[dim] += self.input_patch_shape[dim]
            for idx, dim in enumerate(self.output_patch_dimensions):
                repatched_shape[dim] += self.input_patch_shape[dim]

            padded_input_data = np.zeros(new_input_shape)
            if self.channels_first:
                input_slice = [slice(None)] * 2 + [slice(self.input_patch_shape[dim] // 2, -self.input_patch_shape[dim] // 2, None) for dim in self.patch_dimensions]
            else:
                input_slice = [slice(None)] + [slice(self.input_patch_shape[dim] // 2, -self.input_patch_shape[dim] // 2, None) for dim in self.patch_dimensions] + [slice(None)]
            padded_input_data[tuple(input_slice)] = input_data
            input_data = padded_input_data
        else:
            repatched_shape = self.output_shape

        repatched_image = np.zeros(repatched_shape)

        corner_data_dims = [input_data.shape[axis] for axis in self.patch_dimensions]
        corner_patch_dims = [self.output_patch_shape[axis] for axis in self.patch_dimensions]

        all_corners = np.indices(corner_data_dims)

        # There must be a better way to round up to an integer..
        possible_corners_slice = [slice(None)] + [slice(self.input_patch_shape[dim] // 2, -self.input_patch_shape[dim] // 2, None) for dim in self.patch_dimensions]
        all_corners = all_corners[tuple(possible_corners_slice)]

        for rep_idx in range(self.patch_overlaps):

            if self.verbose:
                docker_print('Predicting patch set', str(rep_idx + 1) + '/' + str(self.patch_overlaps) + '...')

            corners_grid_shape = [slice(None)]
            for dim in range(all_corners.ndim - 1):
                corners_grid_shape += [slice(repetition_offsets[dim][rep_idx], corner_data_dims[dim], corner_patch_dims[dim])]

            corners_list = all_corners[tuple(corners_grid_shape)]
            corners_list = np.reshape(corners_list, (corners_list.shape[0], -1)).T

            if self.check_empty_patch:
                corners_list = self.remove_empty_patches(input_data, corners_list)

            for corner_list_idx in range(0, corners_list.shape[0], self.batch_size):

                corner_batch = corners_list[corner_list_idx:corner_list_idx + self.batch_size]
                input_patches = self.grab_patch(input_data, corner_batch)
                
                prediction = self.run_inference(input_patches)

                self.insert_patch(repatched_image, prediction, corner_batch)

            if rep_idx == 0:
                output_data = np.copy(repatched_image)
            else:
                output_data = self.aggregate_predictions(output_data, repatched_image, rep_idx)

        if self.pad_borders:

            output_slice = [slice(None)] * output_data.ndim  # Weird
            for idx, dim in enumerate(self.output_patch_dimensions):
                # Might not work for odd-shaped patches; check.
                output_slice[dim] = slice(self.input_patch_shape[dim] // 2, -self.input_patch_shape[dim] // 2, 1)
            output_data = output_data[tuple(output_slice)]

        if self.keep_channels is not None:
            output_data = np.take(output_data, self.keep_channels, axis=-1)

        return output_data

    def run_inference(self, data):

        return self.model.predict(data)

    def aggregate_predictions(self, output_data, repatched_image, rep_idx):

        output_data = output_data + (1.0 / (rep_idx)) * (repatched_image - output_data)  # Running Average
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
                output_slice[dim] = slice(corner[idx] - self.input_patch_shape[dim] // 2, corner[idx] + self.input_patch_shape[dim] // 2, 1)

            corner_selections += [np.any(input_data[tuple(output_slice)])]

        return corners_list[corner_selections]

    def grab_patch(self, input_data, corner_list):

        """ Given a corner coordinate, a patch_shape, and some input_data, returns a patch or array of patches.
        """

        output_patches_shape = (corner_list.shape[0], ) + self.input_patch_shape[1:]
        output_patches = np.zeros((output_patches_shape))

        for corner_idx, corner in enumerate(corner_list):
            output_slice = [slice(None)] * input_data.ndim  # Weird
            for idx, dim in enumerate(self.patch_dimensions):
                output_slice[dim] = slice(corner[idx] - self.input_patch_shape[dim] // 2, corner[idx] + self.input_patch_shape[dim] // 2, 1)

            output_patches[corner_idx, ...] = input_data[tuple(output_slice)]

        return output_patches

    def insert_patch(self, input_data, patches, corner_list):

        # Some ineffeciencies in the function. TODO: come back and rewrite.

        for corner_idx, corner in enumerate(corner_list):
            insert_slice = [slice(None)] * input_data.ndim  # Weird
            for idx, dim in enumerate(self.output_patch_dimensions):
                # Might not work for odd-shaped patches; check.
                insert_slice[dim] = slice(corner[idx] - self.output_patch_shape[dim] // 2, corner[idx] + self.output_patch_shape[dim] // 2, 1)

            insert_patch = patches[corner_idx, ...]
            if not np.array_equal(np.take(self.output_patch_shape, self.output_patch_dimensions), np.take(self.input_patch_shape, self.patch_dimensions)):  # Necessary if statement?
                patch_slice = [slice(None)] * insert_patch.ndim  # Weird
                for idx, dim in enumerate(self.output_patch_dimensions):
                    # Might not work for odd-shaped patches; check.
                    patch_slice[dim] = slice((self.input_patch_shape[dim] - self.output_patch_shape[dim]) // 2, -(self.input_patch_shape[dim] - self.output_patch_shape[dim]) // 2, 1)

                insert_patch = insert_patch[tuple(patch_slice)]

            input_data[tuple(insert_slice)] = insert_patch

        return input_data