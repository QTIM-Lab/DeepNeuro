import numpy as np
import matplotlib.pyplot as plt


def check_data(output_data=None, data_collection=None, batch_size=4, merge_batch=True, output_filepath=None, viz_rows=2, viz_mode_3d='2d_center', color_range=None):

    if data_collection is not None:
        generator = data_collection.data_generator(perpetual=True, verbose=False, batch_size=batch_size)
        output_data = next(generator)

    if color_range is None:
        color_range = {label: [np.min(data), np.max(data)] for label, data in output_data.items()}

    fig, axarr = plt.subplots(len(output_data.keys()))

    for plot_idx, (label, data) in enumerate(output_data.items()):
        viz_columns = int(np.ceil(batch_size / float(viz_rows)))
        merged_data = merge_data(data, [viz_rows, viz_columns], data.shape[-1])

        if data.shape[-1] == 3:
            # Weird matplotlib bug:
            if np.min(merged_data) < 0:
                merged_data = (merged_data - np.min(merged_data)) / (np.max(merged_data) - np.min(merged_data))
            plt_image = axarr[plot_idx].imshow(np.squeeze(merged_data), cmap=plt.get_cmap('hot'), vmin=color_range[label][0], vmax=color_range[label][1], interpolation='none')

            fig.colorbar(plt_image, ax=axarr[plot_idx])

        elif data.shape[-1] == 1:
            plt_image = axarr[plot_idx].imshow(np.squeeze(merged_data), cmap='gray', vmin=color_range[label][0], vmax=color_range[label][1], interpolation='none')
            fig.colorbar(plt_image, ax=axarr[plot_idx], cmap='gray')

        axarr[plot_idx].set_title(label)

    plt.show()

    return


def merge_data(images, size, channels=3):

    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], channels))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img