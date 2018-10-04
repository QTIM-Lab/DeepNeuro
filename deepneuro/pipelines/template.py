""" Not implemented. Reads YAML files to create DeepNeuro pipelines.
"""

import yaml


def parse_template(input_file):

    template = yaml.load(open(input_file))

    for key, label in list(template.items()):
        print(key, label)

    ### Process Inputs ###
    print(template['Inputs'])

    return


def create_cli_from_template():

    return


def write_template_to_script():

    return


if __name__ == '__main__':

    parse_template('sample_template.yml')

    pass