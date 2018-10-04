import tables


def hdf5_transpose(hdf5, output_hdf5, axes):

    open_hdf5 = tables.open_file(hdf5, "r")

    for data_group in open_hdf5.root._f_iter_nodes():
        if '_affines' not in data_group.name and '_casenames' not in data_group.name:

            print((data_group.shape))

    return


if __name__ == '__main__':
    hdf5_transpose()