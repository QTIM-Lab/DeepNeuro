import numpy as np


def data_generator(data_collection, data_group_labels=None, perpetual=False, case_list=None, yield_data=True, verbose=False, batch_size=1, return_dict=True):

    data_groups = data_collection.get_data_groups(data_group_labels)

    if case_list is None:
        case_list = data_collection.cases

    # Kind of a funny way to do batches
    if return_dict:
        data_batch = [[] for data_group in data_groups]
    else:
        data_batch = {data_group.label: [] for data_group in data_groups}

    while True:

        np.random.shuffle(case_list)

        for case_idx, case_name in enumerate(case_list):

            if verbose:
                print 'Working on image.. ', case_idx, 'at', case_name

            try:
                data_collection.load_case_data(case_name)
            except KeyboardInterrupt:
                raise
            except:
                print 'Hit error on', case_name, 'skipping.'
                yield False

            recursive_augmentation_generator = data_collection.recursive_augmentation(data_groups, augmentation_num=0)

            for i in xrange(data_collection.multiplier):
                next(recursive_augmentation_generator)

                if yield_data:
                    # TODO: Do this without if-statement and for loop?

                    for data_idx, data_group in enumerate(data_groups):
                        if len(data_collection.augmentations) == 0:
                            data_batch[data_idx].append(data_group.base_case[0])
                        else:
                            data_batch[data_idx].append(data_group.augmentation_cases[-1][0])

                    if len(data_batch[0]) == batch_size:
                        # More strange indexing behavior. Shape inconsistency to be resolved.
                        yield tuple([np.stack(data_list) for data_list in data_batch])
                        data_batch = [[] for data_group in data_groups]

                else:
                    yield True

        if not perpetual:
            yield None
            break