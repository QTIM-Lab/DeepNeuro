from collections import defaultdict

class Preprocessor(object):


    def __init__(self, data_groups=None, channel_dim=-1, save_output=True, **kwargs):

        self.output_shape = None
        self.initialization = False

        self.data_groups = {data_group: None for data_group in data_groups}

        self.preproccesor_string = ''

        self.save_output = save_output
        self.channel_dim = channel_dim

        self.outputs = defaultdict(list)

        self.load(kwargs)

        return

    def load(self, kwargs):

        """ This method is used by children classes to load additional attributes from kwargs. These
            may be parameters specific to a certain model type, for example.
        """

        return

    def execute(self, case):

        for label, data_group in self.data_groups.iteritems():

            data_group.augmentation_cases[augmentation_num+1] = data_group.augmentation_cases[augmentation_num]


    def initialize(self):

        if not self.initialization:
            self.initialization = True

    def reset(self, augmentation_num):
        return

    def append_data_group(self, data_group):
        self.data_groups[data_group.label] = data_group