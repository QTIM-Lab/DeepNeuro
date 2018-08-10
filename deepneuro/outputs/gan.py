import numpy as np

from deepneuro.outputs.output import Output
from deepneuro.utilities.util import add_parameter, docker_print

class GanSLERP(Output):

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

        super(GanSLERP, self).load(kwargs)

        # Patching Parameters
        add_parameter(self, kwargs, 'patch_overlaps', 1)
        add_parameter(self, kwargs, 'output_patch_shape', None)
        add_parameter(self, kwargs, 'patch_overlaps', True)
        add_parameter(self, kwargs, 'check_empty_patch', True)
        add_parameter(self, kwargs, 'pad_borders', True)

        add_parameter(self, kwargs, 'patch_dimensions', None)

        add_parameter(self, kwargs, 'output_patch_dimensions', self.patch_dimensions)

    def process_case(self, input_data, model=None):

        # A little bit strange to access casename this way. Maybe make it an optional
        # return of the generator.

        # Note that input_modalities as the first input is hard-coded here. Very fragile.

        # If an image is being repatched, its output shape is not certain. We attempt to infer it from
        # the input data. This is wonky. Move this to PatchInference, maybe.

        if model is not None:
            self.model = model

        output_data = self.predict(input_data, model)

        # Will fail for time-data.
        if self.channels_first:
            output_data = np.swapaxes(output_data, 1, -1)

        self.return_objects.append(output_data)

        return output_data

    def predict(self, input_data, model=None):

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
            if self.channels_first:
                input_slice = [slice(None)] * 2 + [slice(self.input_patch_shape[dim] / 2, -self.input_patch_shape[dim] / 2, None) for dim in self.patch_dimensions]
            else:
                input_slice = [slice(None)] + [slice(self.input_patch_shape[dim] / 2, -self.input_patch_shape[dim] / 2, None) for dim in self.patch_dimensions] + [slice(None)]
            padded_input_data[tuple(input_slice)] = input_data
            input_data = padded_input_data

        repatched_image = np.zeros(repatched_shape)

        corner_data_dims = [input_data.shape[axis] for axis in self.patch_dimensions]
        corner_patch_dims = [self.output_patch_shape[axis] for axis in self.patch_dimensions]

        all_corners = np.indices(corner_data_dims)

        # There must be a better way to round up to an integer..
        possible_corners_slice = [slice(None)] + [slice(self.input_patch_shape[dim] / 2, -self.input_patch_shape[dim] / 2, None) for dim in self.patch_dimensions]
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
                
                prediction = self.model.predict(input_patches)
                
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
            output_data = output_data[tuple(output_slice)]

        return output_data