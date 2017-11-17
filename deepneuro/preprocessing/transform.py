
import subprocess
import os
import glob

from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.utilities.util import add_parameter, replace_suffix

from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti

class Resample(Preprocessor):

    def load(self, kwargs):

        """ Parameters
            ----------
            depth : int, optional
                Specified the layers deep the proposed U-Net should go.
                Layer depth is symmetric on both upsampling and downsampling
                arms.
            max_filter: int, optional
                Specifies the number of filters at the bottom level of the U-Net.

        """

        add_parameter(self, kwargs, 'command', ['Slicer', '--launch'])

        add_parameter(self, kwargs, 'dimensions', [1,1,1])
        add_parameter(self, kwargs, 'interpolation', 'linear')
        add_parameter(self, kwargs, 'reference_file', None)

        self.interpolation_dict = {'nearestNeighbor': 'nn', 'linear': 'linear'}

        self.preprocessor_string = '_Resampled_' + str(self.dimensions).strip('[]').replace(' ', '').replace(',', '')


    def execute(self, case):

        for label, data_group in self.data_groups.iteritems():

            for index, file in enumerate(data_group.preprocessed_case):

                output_filename = replace_suffix(file, '', self.preprocessor_string)

                if self.reference_file is None:
                    specific_command = self.command + ['ResampleScalarVolume', '-i', self.interpolation, file, output_filename]
                else:
                    specific_command = self.command + ['ResampleScalarVectorDWIVolume', '-R', self.reference_file, '--interpolation', self.interpolation_dict[self.interpolation], file, output_filename]
                
                subprocess.call(' '.join(specific_command), shell=True)

                print 'save_output', self.save_output
                print data_group.preprocessed_case[index]
                print data_group.data[case][index]
                print data_group.preprocessed_case[index] != data_group.data[case][index]
                if not self.save_output and data_group.preprocessed_case[index] != data_group.data[case][index]:
                    os.remove(data_group.preprocessed_case[index])

                data_group.preprocessed_case[index] = output_filename

                self.outputs['outputs'] += [output_filename]

class Coregister(Preprocessor):

    def load(self, kwargs):

        """ Parameters
            ----------
            depth : int, optional
                Specified the layers deep the proposed U-Net should go.
                Layer depth is symmetric on both upsampling and downsampling
                arms.
            max_filter: int, optional
                Specifies the number of filters at the bottom level of the U-Net.

        """


        add_parameter(self, kwargs, 'command', ['Slicer', '--launch', 'BRAINSFit'])

        add_parameter(self, kwargs, 'transform_type', 'Rigid,ScaleVersor3D,ScaleSkewVersor3D,Affine')
        add_parameter(self, kwargs, 'transform_initialization', 'useMomentsAlign')
        add_parameter(self, kwargs, 'interpolation', 'Linear')

        if 'reference_channel' in kwargs:
            self.reference_channel = kwargs.get('reference_channel')
        else:
            self.reference_channel = None

        if 'reference_file' in kwargs:
            self.reference_file = kwargs.get('reference_file')
        else:
            self.reference_file = None

        if 'sampling_percentage' in kwargs:
            self.sampling_percentage = kwargs.get('sampling_percentage')
        else:
            self.sampling_percentage = .06

        self.interpolation_dict = {'nearestNeighbor': 'nn'}

        self.preprocessor_string = '_Registered'


    def execute(self, case):

        for label, data_group in self.data_groups.iteritems():

            for index, file in enumerate(data_group.preprocessed_case):

                output_filename = replace_suffix(file, '', self.preprocessor_string)

                self.outputs['outputs'] += [output_filename]

                if self.reference_channel is not None:
                    if self.reference_channel == index:
                        data_group.preprocessed_case[index] = file
                        continue
                    else:
                        specific_command = self.command + ['--fixedVolume', data_group.preprocessed_case[self.reference_channel], '--transformType', self.transform_type, '--initializeTransformMode', self.transform_initialization, '--interpolationMode', self.interpolation, '--samplingPercentage', str(self.sampling_percentage), '--movingVolume', file, '--outputVolume', output_filename]
                else:
                    specific_command = self.command + ['--fixedVolume', '"' + self.reference_file + '"', '--transformType', self.transform_type, '--initializeTransformMode', self.transform_initialization, '--interpolationMode', self.interpolation, '--samplingPercentage', str(self.sampling_percentage), '--movingVolume', file, '--outputVolume', output_filename]

                subprocess.call(' '.join(specific_command), shell=True)

                if not self.save_output and data_group.preprocessed_case[index] != data_group.data[case][index]:
                    os.remove(data_group.preprocessed_case[index])

                data_group.preprocessed_case[index] = output_filename

                self.outputs['outputs'] += [output_filename]