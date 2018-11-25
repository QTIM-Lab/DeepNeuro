import os

from deepneuro.data.data_collection import DataCollection
from deepneuro.models.model import load_old_model
from deepneuro.load.load import load


def load_data(inputs, output_folder, input_directory=None, ground_truth=None, input_data=None, verbose=True):

    """ In the future, this will need to be modified for multiple types of inputs (i.e. data groups).
    """

    inputs = [os.path.abspath(input_filename) for input_filename in inputs]
    output_folder = os.path.abspath(output_folder)

    input_data = {'input_data': inputs}

    if ground_truth is not None:
        input_data['ground_truth'] = [ground_truth]

    if input_directory is None:

        if any(data is None for data in input_data):
            raise ValueError("Cannot run pipeline; required inputs are missing.")

        data_collection = DataCollection(verbose=verbose)
        data_collection.add_case(input_data, case_name=output_folder)

    else:
        data_collection = DataCollection(input_directory, data_group_dict=input_data, verbose=verbose)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if verbose:
        print('File loading completed.')

    return data_collection


def load_model_with_output(model_path=None, model_name=None, outputs=None, postprocessors=None, **kwargs):

    if model_path is not None:
        model = load_old_model(model_path, **kwargs)

    elif model_name is not None:
        model = load_old_model(load(model_name), **kwargs)

    else:
        print('Error. No model provided.')
        return
    
    for output in outputs:
        model.append_output([output])

        for postprocessor in postprocessors:
            output.append_postprocessor([postprocessor]) 

    return model