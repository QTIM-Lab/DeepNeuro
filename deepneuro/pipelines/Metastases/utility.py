import tables

def switch_channels(hdf5, channel_1, channel_2):

                open_hdf5 = tables.open_file(self.data_storage, "r")

            for data_group in open_hdf5.root._f_iter_nodes():
                if '_affines' not in data_group.name and '_casenames' not in data_group.name:

    return

if __name__ == '__main__':
    switch_channels()
