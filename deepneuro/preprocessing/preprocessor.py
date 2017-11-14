class Preprocessor(object):


    def __init__(self, data_groups=None):

        self.output_shape = None
        self.initialization = False

        self.data_groups = {data_group: None for data_group in data_groups}

        return

    def execute(self, image):

        return image

    def initialize(self):

        if not self.initialization:
            self.initialization = True

    def reset(self, augmentation_num):
        return

    def append_data_group(self, data_group):
        self.data_groups[data_group.label] = data_group