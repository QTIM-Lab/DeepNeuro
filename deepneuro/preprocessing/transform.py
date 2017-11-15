
import subprocess
import os
import glob

from deepneuro.preprocessing.preprocessor import Preprocessor

from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti
from qtim_tools.qtim_utilities.file_util import replace_suffix, nifti_splitext

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

        if 'command' in kwargs:
            self.command = kwargs.get('command')
        else:
            self.command = ['Slicer', '--launch']

        if 'dimensions' in kwargs:
            self.dimensions = kwargs.get('dimensions')
        else:
            self.dimensions = [1,1,1]

        if 'interpolation' in kwargs:
            self.interpolation = kwargs.get('interpolation')
        else:
            self.interpolation = 'linear'

        if 'reference_file' in kwargs:
            self.reference_file = kwargs.get('reference_file')
        else:
            self.reference_file = None

        self.interpolation_dict = {'nearestNeighbor': 'nn', 'linear': 'linear'}

        self.preprocessor_string = '_Resampled_' + str(self.dimensions).strip('[]').replace(' ', '').replace(',', '')


    def execute(self, case):

        print 'ABOUT TO RESAMPLE'

        for label, data_group in self.data_groups.iteritems():

            for index, file in enumerate(data_group.preprocessed_case):

                output_filename = replace_suffix(file, '', self.preprocessor_string)

                if self.reference_file is None:
                    specific_command = self.command + ['ResampleScalarVolume', '-i', self.interpolation, file, output_filename]
                else:
                    specific_command = self.command + ['ResampleScalarVectorDWIVolume', '-R', self.reference_file, '--interpolation', self.interpolation_dict[self.interpolation], file, output_filename]
                
                # subprocess.call(' '.join(specific_command), shell=True)

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

        if 'command' in kwargs:
            self.command = kwargs.get('command')
        else:
            self.command = ['Slicer', '--launch', 'BRAINSFit']

        if 'transform_type' in kwargs:
            self.transform_type = kwargs.get('transform_type')
        else:
            self.transform_type = 'Rigid,ScaleVersor3D,ScaleSkewVersor3D,Affine'

        if 'transform_initialization' in kwargs:
            self.transform_initialization = kwargs.get('transform_initialization')
        else:
            self.transform_initialization = 'useMomentsAlign'

        if 'interpolation' in kwargs:
            self.interpolation = kwargs.get('interpolation')
        else:
            self.interpolation = 'Linear'

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