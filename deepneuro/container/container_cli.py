import os

from subprocess import call

from deepneuro.utilities.util import quotes


def docker_wrapper(**kwargs):

    return


def nvidia_docker_wrapper(command, cli_args=None, filename_args=None, interactive=False, docker_container='deepneuro'):

    if filename_args is not None:
        filename_args = [arg for arg in filename_args if cli_args[arg] is not None]
        mounted_dir = os.path.abspath(os.path.dirname(os.path.commonprefix([cli_args[arg] for arg in filename_args])))
        for arg in filename_args:
            after_mounted = os.path.abspath(cli_args[arg]).split(mounted_dir, 1)[1].lstrip(os.sep)
            cli_args[arg] = quotes(os.path.join('/INPUT_DATA', after_mounted))
    else:
        pass  # TODO: Default behavior when mounted directory not needed.

    if interactive:
        docker_command = ['nvidia-docker', 'run', '-it', '-v', mounted_dir + ':/INPUT_DATA', docker_container, 'bash']

    else:
        docker_command = ['nvidia-docker', 'run', '--rm', '-v', mounted_dir + ':/INPUT_DATA', docker_container] + command

        # This presumes everything is an optional arg, which is wrong.
        # Rewrite this to accept command line strings, instead of Python variables.
        # Would solve problem..
        for arg in cli_args:
            if cli_args[arg] is True:
                docker_command += ['-' + str(arg)]
            elif not cli_args[arg] or cli_args[arg] is None:
                continue
            else:
                docker_command += ['-' + str(arg) + ' ' + cli_args[arg]]

    call(' '.join(docker_command), shell=True)


def singularity_wrapper(command, cli_args=None, filename_args=None, interactive=False, container='deepneuro'):

    raise NotImplementedError

    if filename_args is not None:
        filename_args = [arg for arg in filename_args if cli_args[arg] is not None]
        mounted_dir = os.path.abspath(os.path.dirname(os.path.commonprefix([cli_args[arg] for arg in filename_args])))
        for arg in filename_args:
            after_mounted = os.path.abspath(cli_args[arg]).split(mounted_dir, 1)[1].lstrip(os.sep)
            cli_args[arg] = quotes(os.path.join('/INPUT_DATA', after_mounted))
    else:
        pass  # TODO: Default behavior when mounted directory not needed.

    if interactive:
        docker_command = ['nvidia-docker', 'run', '-it', '-v', mounted_dir + ':/INPUT_DATA', container, 'bash']

    else:
        docker_command = ['nvidia-docker', 'run', '--rm', '-v', mounted_dir + ':/INPUT_DATA', container] + command

        # This presumes everything is an optional arg, which is wrong.
        # Rewrite this to accept command line strings, instead of Python variables.
        # Would solve problem..
        for arg in cli_args:
            if cli_args[arg] is True:
                docker_command += ['-' + str(arg)]
            elif not cli_args[arg] or cli_args[arg] is None:
                continue
            else:
                docker_command += ['-' + str(arg) + ' ' + cli_args[arg]]

    call(' '.join(docker_command), shell=True)