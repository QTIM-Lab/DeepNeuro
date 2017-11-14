class Output(object):


    def __init__(self, data_collection=None, inputs=['input_modalities'], output_directory=None, output_filename='prediction.nii.gz', batch_size=32, verbose=True, replace_existing=True, case=None, **kwargs):

        self.data_collection = data_collection
        self.inputs = inputs

        self.output_directory = output_directory
        self.output_filename = output_filename

        self.batch_size = batch_size

        self.replace_existing = replace_existing
        self.verbose = verbose
        self.case = case

        self.load(kwargs)

        return

    def load(self, kwargs):

        """ This method is used by children classes to load additional attributes from kwargs. These
            may be parameters specific to a certain model type, for example.
        """

        return

    def execute(self, model):

        return None

