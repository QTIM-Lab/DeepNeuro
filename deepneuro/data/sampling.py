import numpy as np

from keras.utils import Sequence
from pprint import pprint

from deepneuro.core import add_parameter


def data_generator(data_collection, data_group_labels=None, perpetual=False, case_list=None, yield_data=True, verbose=False, batch_size=1, just_one_batch=False):

    data_groups = data_collection.get_data_groups(data_group_labels)

    if case_list is None:
        case_list = data_collection.cases

    if case_list is None or case_list == '':
        print('No cases found. Yielding None.')
        yield None

    data_batch_labels = ['casename']
    for data_group in data_groups:
        data_batch_labels += [data_group.label, data_group.label + '_augmentation_string', data_group.label + '_affine']

    data_batch = {label: [] for label in data_batch_labels}

    while True:

        np.random.shuffle(case_list)

        for case_idx, case_name in enumerate(case_list):

            if data_collection.source == 'hdf5':
                case_name_string = data_groups[0].data_casenames[case_name][0].decode("utf-8")
            else:
                case_name_string = case_name

            if verbose:
                print(('Working on image.. ', case_idx, 'at', case_name_string))

            # Is error-catching useful here?
            if True:
            # try:
                data_collection.load_case_data(case_name)
            # except KeyboardInterrupt:
                # raise
            # except:
                # print 'Hit error on', case_name, 'skipping.'
                # yield False

            recursive_augmentation_generator = data_collection.recursive_augmentation(data_groups, augmentation_num=0)

            for i in range(data_collection.multiplier):
                next(recursive_augmentation_generator)

                if yield_data:
                    # TODO: This section is terribly complex and repetitive. Revise!

                    for data_idx, data_group in enumerate(data_groups):

                        if len(data_collection.augmentations) == 0:
                            data_batch[data_group.label].append(data_group.preprocessed_case[0])
                        else:
                            data_batch[data_group.label].append(data_group.augmentation_cases[-1][0])

                        data_batch[data_group.label + '_augmentation_string'].append(data_group.augmentation_strings[-1])
                        data_batch[data_group.label + '_affine'].append(data_group.preprocessed_affine)

                    data_batch['casename'].append(case_name_string)

                    if len(data_batch[data_groups[0].label]) == batch_size:
                        
                        for label in data_batch:
                            data_batch[label] = np.stack(data_batch[label])

                        if just_one_batch:
                            while True:
                                yield data_batch
                        else:
                            yield data_batch

                        data_batch = {label: [] for label in data_batch_labels}   

                else:
                    yield True

        if not perpetual:
            yield None
            break


def equal_sampling_data_generator(data_collection, data_group_labels=None, perpetual=False, case_list=None, yield_data=True, verbose=False, batch_size=1, just_one_batch=False, class_group=['ground_truth']):

    class_data_group = data_collection.get_data_groups(class_group)[0]
    data_groups = data_collection.get_data_groups(data_group_labels)

    if case_list is None:
        case_list = data_collection.cases

    if case_list is None or case_list == '':
        print('No cases found. Yielding None.')
        yield None

    data_batch_labels = ['casename']
    for data_group in data_groups:
        data_batch_labels += [data_group.label, data_group.label + '_augmentation_string', data_group.label + '_affine']

    data_batch = {label: [] for label in data_batch_labels}

    np.random.shuffle(case_list)
    case_list = np.array(case_list)

    classes = class_data_group.metadata['classes']
    class_indexes = {val: case_list[class_data_group.metadata['distribution_indexes'][val]] for val in classes}
    class_counters = {val: 0 for val in classes}
    class_batch_size = batch_size // len(classes)

    while True:

        for current_class in classes:

            # I'm going to hell for writing code like this
            class_case_list = class_indexes[current_class]
            class_batch = class_case_list[class_counters[current_class]:class_counters[current_class] + class_batch_size]

            class_counters[current_class] += class_batch_size
            if class_counters[current_class] > len(class_case_list):
                class_counters[current_class] = 0
                np.random.shuffle(class_indexes[current_class])

            for case_idx, case_name in enumerate(class_batch):

                if data_collection.source == 'hdf5':
                    case_name_string = data_groups[0].data_casenames[case_name][0].decode("utf-8")
                else:
                    case_name_string = case_name

                if verbose:
                    print(('Working on image.. ', case_idx, 'at', case_name_string))

                data_collection.load_case_data(case_name)
                recursive_augmentation_generator = data_collection.recursive_augmentation(data_groups, augmentation_num=0)

                for i in range(data_collection.multiplier):
                    next(recursive_augmentation_generator)

                    if yield_data:
                        # TODO: This section is terribly complex and repetitive. Revise!

                        for data_idx, data_group in enumerate(data_groups):

                            if len(data_collection.augmentations) == 0:
                                data_batch[data_group.label].append(data_group.preprocessed_case[0])
                            else:
                                data_batch[data_group.label].append(data_group.augmentation_cases[-1][0])

                            data_batch[data_group.label + '_augmentation_string'].append(data_group.augmentation_strings[-1])
                            data_batch[data_group.label + '_affine'].append(data_group.preprocessed_affine)

                        data_batch['casename'].append(case_name_string)

                        if len(data_batch[data_groups[0].label]) == batch_size:
                            
                            for label in data_batch:
                                data_batch[label] = np.stack(data_batch[label])

                            if just_one_batch:
                                while True:
                                    yield data_batch
                            else:
                                yield data_batch

                            data_batch = {label: [] for label in data_batch_labels}   

                    else:
                        yield True

            if not perpetual:
                yield None
                break


class KerasSequence(Sequence):

    def __init__(self, **kwargs):

        add_parameter(self, kwargs, 'batch_size', 32)
        add_parameter(self, kwargs, 'data_collection', None)
        add_parameter(self, kwargs, 'data_group_labels', None)
        add_parameter(self, kwargs, 'case_list', None)
        add_parameter(self, kwargs, 'verbose', False)
        add_parameter(self, kwargs, 'just_one_batch', False)  # Not implemented.
        add_parameter(self, kwargs, 'x_group', ['input_data'])
        add_parameter(self, kwargs, 'y_group', ['ground_truth'])

        self.x_group = self.data_collection.get_data_groups(self.x_group)[0]
        self.y_group = self.data_collection.get_data_groups(self.y_group)[0]
        self.data_groups = self.data_collection.get_data_groups(self.data_group_labels)

        if self.case_list is None:
            self.case_list = self.data_collection.cases

        if self.case_list is None or self.case_list == '':
            print('No cases found. Yielding None.')

        self.case_list = np.array(self.case_list)

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.case_list) / self.batch_size))

    def __getitem__(self, index):

        x_batch, y_batch = [], []
        counter = index * self.batch_size % (len(self.case_list) - self.batch_size)

        for case_idx, case_name in enumerate(self.case_list[counter:counter + self.batch_size]):

            if self.data_collection.source == 'hdf5':
                case_name_string = self.data_groups[0].data_casenames[case_name][0].decode("utf-8")
            else:
                case_name_string = case_name

            if self.verbose:
                print(('Working on image.. ', case_idx, 'at', case_name_string))

            self.data_collection.load_case_data(case_name)
            recursive_augmentation_generator = self.data_collection.recursive_augmentation(self.data_groups, augmentation_num=0)

            for i in range(self.data_collection.multiplier):
                next(recursive_augmentation_generator)

                if len(self.data_collection.augmentations) == 0:
                    x_batch.append(self.x_group.preprocessed_case[0])
                    y_batch.append(self.y_group.preprocessed_case[0])
                else:
                    x_batch.append(self.x_group.augmentation_cases[-1][0])
                    y_batch.append(self.y_group.preprocessed_case[0])

            if len(x_batch) == self.batch_size:

                return np.stack(x_batch), np.stack(y_batch)
    
    def on_epoch_end(self):

        np.random.shuffle(self.case_list)


class KerasSequence_EqualSampling(KerasSequence):

    def __init__(self, **kwargs):

        add_parameter(self, kwargs, 'batch_size', 32)
        add_parameter(self, kwargs, 'class_group', ['ground_truth'])
        add_parameter(self, kwargs, 'data_collection', None)
        add_parameter(self, kwargs, 'data_group_labels', None)
        add_parameter(self, kwargs, 'case_list', None)
        add_parameter(self, kwargs, 'verbose', False)
        add_parameter(self, kwargs, 'just_one_batch', False)  # Not implemented.
        add_parameter(self, kwargs, 'x_group', ['input_data'])
        add_parameter(self, kwargs, 'y_group', ['ground_truth'])

        self.x_group = self.data_collection.get_data_groups(self.x_group)[0]
        self.y_group = self.data_collection.get_data_groups(self.y_group)[0]
        self.class_data_group = self.data_collection.get_data_groups(self.class_group)[0]
        self.data_groups = self.data_collection.get_data_groups(self.data_group_labels)
        self.classes = self.class_data_group.metadata['classes']

        if self.case_list is None:
            self.case_list = self.data_collection.cases

        if self.case_list is None or self.case_list == '':
            print('No cases found. Yielding None.')

        self.total_classes = {idx + 1: 0 for idx in range(4)}
        self.case_list = np.array(self.case_list)

        self.on_epoch_end()

    # @profile
    def __getitem__(self, index):

        x_batch, y_batch = [], []
        counter = index * self.class_batch_size % (len(self.case_list) - self.batch_size)

        for current_class in self.classes:

            # I'm going to hell for writing code like this
            class_case_list = self.class_indexes[current_class]
            class_counter = counter % ((len(class_case_list) - self.class_batch_size))
            class_batch = class_case_list[class_counter:class_counter + self.class_batch_size]

            for case_idx, case_name in enumerate(class_batch):

                self.total_classes[int(current_class)] += 1

                if self.data_collection.source == 'hdf5':
                    case_name_string = self.data_groups[0].data_casenames[case_name][0].decode("utf-8")
                else:
                    case_name_string = case_name

                if self.verbose:
                    print(('Working on image.. ', case_idx, 'at', case_name_string))

                self.data_collection.load_case_data(case_name)
                recursive_augmentation_generator = self.data_collection.recursive_augmentation(self.data_groups, augmentation_num=0)

                for i in range(self.data_collection.multiplier):
                    next(recursive_augmentation_generator)

                    if len(self.data_collection.augmentations) == 0:
                        x_batch.append(self.x_group.preprocessed_case[0])
                        y_batch.append(self.y_group.preprocessed_case[0])
                    else:
                        x_batch.append(self.x_group.augmentation_cases[-1][0])
                        y_batch.append(self.y_group.preprocessed_case[0])

                if len(x_batch) == self.batch_size:

                    return np.stack(x_batch), np.stack(y_batch)
    
    def on_epoch_end(self):

        self.class_indexes = {val: self.case_list[self.class_data_group.metadata['distribution_indexes'][val]] for val in self.classes}
        for key, item in self.class_indexes.items():
            np.random.shuffle(item)

        self.class_counters = {val: 0 for val in self.classes}
        self.class_batch_size = self.batch_size // len(self.classes)