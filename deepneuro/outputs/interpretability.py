import numpy as np
import tables
import os
import umap
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.ndimage as ndimage

from sklearn.cluster import KMeans

from deepneuro.outputs.output import Output
from deepneuro.utilities.util import add_parameter, docker_print, replace_suffix
from deepneuro.utilities.conversion import save_data


class PatchInterpretability(Output):

    def load(self, kwargs):

        super(PatchInterpretability, self).load(kwargs)

        # Model Parameters
        add_parameter(self, kwargs, 'input_channels', None)
        add_parameter(self, kwargs, 'channels_dim', None)
        add_parameter(self, kwargs, 'output_channels', None)
        add_parameter(self, kwargs, 'output_layers', [None])

        # Patching Parameters
        add_parameter(self, kwargs, 'patch_method', 'random')
        add_parameter(self, kwargs, 'patch_num', 50000)
        add_parameter(self, kwargs, 'mask_condition', 0)
        add_parameter(self, kwargs, 'patch_overlaps', 1)
        add_parameter(self, kwargs, 'output_patch_shape', None)
        add_parameter(self, kwargs, 'check_empty_patch', True)
        add_parameter(self, kwargs, 'pad_borders', True)
        add_parameter(self, kwargs, 'patch_dimensions', None)
        add_parameter(self, kwargs, 'output_patch_dimensions', self.patch_dimensions)

        # Aggregation Parameters
        add_parameter(self, kwargs, 'aggregation_method', 'flatten')
        # add_parameter(self, kwargs, 'iterative_aggregation', True)

        # Umap Parameters
        add_parameter(self, kwargs, 'umap_components', 2)

        # Clustering Parameters
        add_parameter(self, kwargs, 'cluster_individual_case', False)
        add_parameter(self, kwargs, 'cluster_num', 10)

        # Output Parameters
        add_parameter(self, kwargs, 'generate_patches', False)
        add_parameter(self, kwargs, 'aggregate_patches', True)
        add_parameter(self, kwargs, 'umap_patches', True)
        add_parameter(self, kwargs, 'cluster_patches', True)
        add_parameter(self, kwargs, 'show_umap_plot', False)
        add_parameter(self, kwargs, 'save_projected_output', True)

        # Output Naming Paramters
        add_parameter(self, kwargs, 'patch_filename', None)
        add_parameter(self, kwargs, 'aggregate_patch_filename', None)
        add_parameter(self, kwargs, 'features_filename', None)
        add_parameter(self, kwargs, 'clusters_filename', None)
        add_parameter(self, kwargs, 'umap_plot_filename', None)
        add_parameter(self, kwargs, 'data_filename', None)
        add_parameter(self, kwargs, 'label_filename', None)

        # Overwrite Parameters
        add_parameter(self, kwargs, 'overwrite_patches', False)
        add_parameter(self, kwargs, 'overwrite_aggregate', True)
        add_parameter(self, kwargs, 'overwrite_features', True)
        add_parameter(self, kwargs, 'overwrite_clusters', True)
        add_parameter(self, kwargs, 'overwrite_plot', True)
        add_parameter(self, kwargs, 'overwrite_outputs', True)

        # Experimental Parameters
        add_parameter(self, kwargs, 'baseline_mean_intensity', False)

        self.open_hdf5_file = None

    # @profile
    def process_case(self, input_data, model=None):

        # A little bit strange to access casename this way. Maybe make it an optional
        # return of the generator.

        # Note that input_modalities as the first input is hard-coded here. Very fragile.

        # If an image is being repatched, its output shape is not certain. We attempt to infer it from
        # the input data. This is wonky. Move this to PatchInference, maybe.

        if model is not None:
            self.model = model

        if self.channels_first:
            input_data = np.swapaxes(input_data, 1, -1)

        if self.input_channels is not None:
            input_data = np.take(input_data, self.input_channels, self.channels_dim)

        # Determine patch shape. Currently only extends to spatial patching.
        # This leading dims business has got to have a better solution..
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

        for layer in self.output_layers:

            self.current_layer = layer
            self.create_output_filenames()

            save_data(input_data[0, ..., 0], self.current_data_filename)

            if self.current_patch_filename is not None:
                if not os.path.exists(self.current_patch_filename) or self.overwrite_patches:
                    self.create_patch_hdf5_file()
                    patch_data = self.generate_patch_data(input_data, model)
                    print(patch_data)

                if self.cluster_individual_case:
                    self.cluster_patch_data(input_data)

            if self.open_hdf5_file is not None:
                self.open_hdf5_file.close()

        return None

    def create_output_filenames(self):

        if self.current_layer is None:
            suffix = '_prediction'
        else:
            suffix = '_' + str(self.current_layer)

        self.current_output_directory = os.path.join(self.output_directory, os.path.basename(self.data_collection.get_current_casename()))
        if not os.path.exists(self.current_output_directory):
            os.mkdir(self.current_output_directory)

        self.current_patch_filename = os.path.join(self.current_output_directory, replace_suffix(self.patch_filename, '', suffix))
        self.current_aggregate_patch_filename = os.path.join(self.current_output_directory, replace_suffix(self.aggregate_patch_filename, '', suffix))
        self.current_umap_feature_output = os.path.join(self.current_output_directory, replace_suffix(self.features_filename, '', suffix))
        self.current_umap_cluster_output = os.path.join(self.current_output_directory, replace_suffix(self.clusters_filename, '', suffix))
        self.current_umap_plot_filename = os.path.join(self.current_output_directory, replace_suffix(self.umap_plot_filename, '', suffix))
        self.current_data_filename = os.path.join(self.current_output_directory, self.data_filename)
        self.current_label_filename = os.path.join(self.current_output_directory, replace_suffix(self.label_filename, '', suffix))

        return

    def cluster_patch_data(self, input_data):

        if self.open_hdf5_file is not None:
            self.open_hdf5_file.close()
        
        open_hdf5 = tables.open_file(self.current_patch_filename, "r")
        output_npy_file = replace_suffix(self.current_aggregate_patch_filename, '', '_' + self.aggregation_method)

        # Load and aggregate data for analysis.
        if not os.path.exists(output_npy_file) or self.overwrite_aggregate:
            patch_data = self.aggregate_patch_data(open_hdf5, output_npy_file)
        else:
            patch_data = np.load(output_npy_file)

        print(patch_data.shape)
        print(input_data.shape)

        # Calculate Features and Clusters
        if self.baseline_mean_intensity:
            umap_features = patch_data
        else:
            if not os.path.exists(self.current_umap_feature_output) or self.overwrite_features:
                umap_features = umap.UMAP(n_neighbors=30, min_dist=0.0, verbose=True).fit_transform(patch_data)
                np.save(self.current_umap_feature_output, umap_features)
            else:
                umap_features = np.load(self.current_umap_feature_output)

        if not os.path.exists(self.current_umap_cluster_output) or self.overwrite_clusters:
            k_clusters = KMeans(n_clusters=self.cluster_num).fit_predict(umap_features)
            np.save(self.current_umap_cluster_output, k_clusters)
        else:
            k_clusters = np.load(self.current_umap_cluster_output)

        # Plot UMAP and Save Output
        if not self.baseline_mean_intensity:
            if not os.path.exists(self.current_umap_plot_filename) or self.overwrite_plot:
                self.plot_umap(umap_features, clusters=k_clusters, show_plot=self.show_umap_plot, output_filename=self.current_umap_plot_filename)

        # Map Back to Original Data
        corners = open_hdf5.root.corners
        input_data = input_data[0, ..., 0]
        output_array = np.zeros_like(input_data)
        print(output_array.shape)
        print(corners.shape)
        for idx, coordinate in enumerate(corners):
            # print(coordinate)
            output_array[int(coordinate[0]), int(coordinate[1]), int(coordinate[2])] = k_clusters[idx] + 1
        padded_points = ndimage.maximum_filter(output_array, 3)
        padded_points[output_array != 0] = output_array[output_array != 0]

        save_data(padded_points, self.current_label_filename)

        return

    def plot_umap(self, data, show_plot=True, output_filename=None, clusters=None, title='UMAP Clustering'):

        slicer_cmap = mpl.colors.ListedColormap(['red', 'green', 'blue', 'cyan'])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data[:, 0], data[:, 1], c=clusters, cmap=slicer_cmap)
        plt.title(title, fontsize=18)

        if output_filename is not None:
            plt.savefig(self.current_umap_plot_filename, bbox_inches='tight')

        if show_plot:
            plt.show()

        plt.clf()

    def aggregate_patch_data(self, hdf5, output_npy_file):

        patch_data = hdf5.root.patches

        if self.aggregation_method == 'flatten':
            patch_data = np.reshape(patch_data, (patch_data.shape[0], -1))
        elif self.aggregation_method == 'average':
            if self.iterative_aggregation:
                new_patch_data = None
                for i in range(patch_data.shape[-1]):
                    print(i)
                    patch_data_slice = patch_data[..., i]
                    mean_slice = np.mean(patch_data_slice, axis=range(1, len(patch_data_slice.shape)))
                    print(mean_slice.shape)
                    if new_patch_data is None:
                        new_patch_data = mean_slice
                    else:
                        new_patch_data = np.concatenate((new_patch_data, mean_slice), axis=1)
                patch_data = new_patch_data
            else:
                patch_data = np.mean(patch_data, axis=tuple(range(1, len(patch_data.shape) - 1)))
        else:
            raise NotImplementedError

        np.save(output_npy_file, patch_data)

        return patch_data

    # @profile
    def generate_patch_data(self, input_data, model=None):

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

        corner_data_dims = [input_data.shape[axis] for axis in self.patch_dimensions]

        if self.patch_method == 'random':

            if self.check_empty_patch:
                all_corners = np.array(np.nonzero(input_data))[1:-1]
            else:
                all_corners = np.indices(corner_data_dims)
                # There must be a better way to round up to an integer..
                possible_corners_slice = [slice(None)] + [slice(self.input_patch_shape[dim] // 2, -self.input_patch_shape[dim] // 2, None) for dim in self.patch_dimensions]
                all_corners = all_corners[tuple(possible_corners_slice)]

            all_corners = np.reshape(all_corners, (all_corners.shape[0], -1)).T
            corner_selection = all_corners[np.random.choice(all_corners.shape[0], size=self.patch_num, replace=False), :]

            if self.current_layer is not None:
                output_operation = self.model.get_layer_output_function(self.current_layer)

            for corner_list_idx in range(0, corner_selection.shape[0], self.batch_size):

                print(corner_list_idx, '/', corner_selection.shape[0])

                corner_batch = corner_selection[corner_list_idx:corner_list_idx + self.batch_size]
                input_patches = self.grab_patch(input_data, corner_batch)

                if self.baseline_mean_intensity:
                    prediction = input_patches
                elif self.current_layer is None:
                    prediction = self.model.predict(input_patches)
                else:
                    prediction = output_operation([input_patches])[0]

                prediction = np.mean(prediction, axis=(1, 2, 3), keepdims=True)

                depadded_corner_batch = np.zeros_like(corner_batch)
                for corner_idx, corner in enumerate(corner_batch):
                    depadded_corner_batch[corner_idx] = [corner[dim] - self.input_patch_shape[dim + 1] // 2 for dim in range(corner_batch.shape[1])]
                # prediction = self.model.predict(input_patches)
                    
                self.write_patches_to_hdf5(prediction, depadded_corner_batch)

            pass

        elif self.patch_method == 'grid':

            repetition_offsets = [np.linspace(0, self.input_patch_shape[axis] - 1, self.patch_overlaps, dtype=int) for axis in self.patch_dimensions]
            repatched_image = np.zeros(repatched_shape)

            corner_patch_dims = [self.output_patch_shape[axis] for axis in self.patch_dimensions]

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
                    
                    print(prediction.shape)
                    print(corner_batch)

                if rep_idx == 0:
                    output_data = np.copy(repatched_image)
                else:
                    output_data = output_data + (1.0 / (rep_idx)) * (repatched_image - output_data)  # Running Average

        # if self.pad_borders:

        #     output_slice = [slice(None)] * output_data.ndim  # Weird
        #     for idx, dim in enumerate(self.output_patch_dimensions):
        #         # Might not work for odd-shaped patches; check.
        #         output_slice[dim] = slice(self.input_patch_shape[dim] // 2, -self.input_patch_shape[dim] // 2, 1)
        #     output_data = output_data[tuple(output_slice)]

        return

    def pad_image(self):

        return

    def get_output_shape(self):

        return

    def create_patch_hdf5_file(self):

        self.open_hdf5_file = tables.open_file(self.current_patch_filename, mode='w')
        filters = tables.Filters(complevel=5, complib='blosc')

        num_cases = self.patch_num * self.data_collection.total_cases

        if num_cases == 0:
            raise Exception('WARNING: No cases found. Cannot write to file.')

        if self.current_layer is None:
            data_shape = list(self.output_patch_shape)
        else:
            data_shape = list(self.model.get_layer_output_shape(self.current_layer))
        data_shape[0:4] = [0, 1, 1, 1]
        data_shape = tuple(data_shape)

        # for output_name, output_shape in self.output_layers
        self.output_storage = self.open_hdf5_file.create_earray(self.open_hdf5_file.root, 'patches', tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=num_cases)

        # Naming convention is bad here, TODO, think about this.
        self.casename_storage = self.open_hdf5_file.create_earray(self.open_hdf5_file.root, 'casenames', tables.StringAtom(256), shape=(0, 1), filters=filters, expectedrows=num_cases)
        self.corner_storage = self.open_hdf5_file.create_earray(self.open_hdf5_file.root, 'corners', tables.Float32Atom(), shape=(0, 3), filters=filters, expectedrows=num_cases)

        return self.open_hdf5_file

    def write_patches_to_hdf5(self, patches, corners):

        self.output_storage.append(patches)
        self.corner_storage.append(corners)

        # if self.data_collection.source == 'hdf5':
        #     self.casename_storage.append(np.array(self.data_collection.data_groups[self.inputs[0].data_casenames[case_name][0])[np.newaxis][np.newaxis])
        # else:
        #     self.casename_storage.append(np.array(bytes(self.data_collection.current_case, 'utf-8'))[np.newaxis][np.newaxis])

        pass

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


class AggregatePatchInterpretability(PatchInterpretability):

    def load(self, kwargs):

        super(AggregatePatchInterpretability, self).load(kwargs)
