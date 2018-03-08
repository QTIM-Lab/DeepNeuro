
import subprocess
import os
import sys
import glob

from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.utilities.util import add_parameter, replace_suffix
from deepneuro.utilities.conversion import save_numpy_2_nifti

FNULL = open(os.devnull, 'w')

class Resample(Preprocessor):

    def load(self, kwargs):

        add_parameter(self, kwargs, 'command', ['Slicer', '--launch'])

        add_parameter(self, kwargs, 'dimensions', [1,1,1])
        add_parameter(self, kwargs, 'interpolation', 'linear')
        add_parameter(self, kwargs, 'reference_file', None)

        add_parameter(self, kwargs, 'name', 'Resample')
        add_parameter(self, kwargs, 'preprocessor_string', '_Resampled_' + str(self.dimensions).strip('[]').replace(' ', '').replace(',', ''))

        self.interpolation_dict = {'nearestNeighbor': 'nn', 'linear': 'linear'}
        self.dimensions = str(self.dimensions).strip('[]').replace(' ', '')

    def preprocess(self):

        if self.reference_file is None:
            specific_command = self.command + ['ResampleScalarVolume', '-i', self.interpolation, '-s', self.dimensions, self.base_file, self.output_filename]
        else:
            specific_command = self.command + ['ResampleScalarVectorDWIVolume', '-R', self.reference_file, '--interpolation', self.interpolation_dict[self.interpolation], self.base_file, self.output_filename]
        
        subprocess.call(' '.join(specific_command), shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

class Coregister(Preprocessor):

    def load(self, kwargs):

        add_parameter(self, kwargs, 'command', ['Slicer', '--launch', 'BRAINSFit'])

        add_parameter(self, kwargs, 'transform_type', 'Rigid,ScaleVersor3D,ScaleSkewVersor3D,Affine')
        add_parameter(self, kwargs, 'transform_initialization', 'useMomentsAlign')
        add_parameter(self, kwargs, 'interpolation', 'Linear')
        add_parameter(self, kwargs, 'sampling_percentage', .06)

        add_parameter(self, kwargs, 'reference_channel', None)
        add_parameter(self, kwargs, 'reference_file', None)

        add_parameter(self, kwargs, 'name', 'Registration')
        add_parameter(self, kwargs, 'preprocessor_string', '_Registered')

        self.interpolation_dict = {'nearestNeighbor': 'nn'}

    def execute(self, case):

        """ There is a lot of repeated code in the preprocessors. Think about preprocessor structures and work on this class.
        """

        self.initialize() # TODO: make overwrite work with initializations

        for label, data_group in self.data_groups.iteritems():

            for index, file in enumerate(data_group.preprocessed_case):

                if self.verbose:
                    print 'Preprocessor: ', self.name, '. Case: ', file
                    sys.stdout.flush()

                self.base_file = file # Weird name for this, make more descriptive

                if self.output_folder is None:
                    self.output_filename = replace_suffix(file, '', self.preprocessor_string)
                else:
                    self.output_filename = os.path.join(self.output_folder, os.path.basename(replace_suffix(file, '', self.preprocessor_string)))

                if self.reference_channel is not None:
                    if self.reference_channel == index:
                        data_group.preprocessed_case[index] = self.base_file
                        continue
                    else:
                        specific_command = self.command + ['--fixedVolume', data_group.preprocessed_case[self.reference_channel], '--transformType', self.transform_type, '--initializeTransformMode', self.transform_initialization, '--interpolationMode', self.interpolation, '--samplingPercentage', str(self.sampling_percentage), '--movingVolume', self.base_file, '--outputVolume', self.output_filename]
                else:
                    specific_command = self.command + ['--fixedVolume', '"' + self.reference_file + '"', '--transformType', self.transform_type, '--initializeTransformMode', self.transform_initialization, '--interpolationMode', self.interpolation, '--samplingPercentage', str(self.sampling_percentage), '--movingVolume', self.base_file, '--outputVolume', self.output_filename]

                subprocess.call(' '.join(specific_command), shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

                if not self.save_output and data_group.preprocessed_case[index] != data_group.data[case][index]:
                    os.remove(data_group.preprocessed_case[index])

                data_group.preprocessed_case[index] = self.output_filename

                # Outputs is broken for multiple data groups.
                self.outputs['outputs'] += [self.output_filename]