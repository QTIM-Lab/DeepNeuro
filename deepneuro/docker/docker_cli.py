import os

def nvidia_docker_wrapper(command, cli_args=None, filename_args=None, interactive=True, docker_container='deepneuro'):

    print args

    filename_args = [arg for arg in filename_args if arg is not None]
    mounted_dir = os.path.abspath(os.path.dirname(os.path.commonprefix([getattr(cli_args, arg) for arg in filename_args])))
    for arg in filename_args:
        setattr(cli_args, arg, os.path.abspath(cli_args.arg).split(mounted_dir,1)[1][1:])

    if hasattr(cli_args, 'gpu_num'):
        if getattr(cli_args, 'gpu_num') is None:
            setattr(cli_args, 'gpu_num', 0)
        else:
            setattr(cli_args, 'gpu_num', str(cli_args.gpu_num))

    if interactive:
        docker_command = ['nvidia-docker', 'run', '-it', '-v', mounted_dir + ':/INPUT_DATA', docker_container, 'bash']

    else:
        docker_command = ['nvidia-docker', 'run', '--rm', '-v', mounted_dir + ':/INPUT_DATA', docker_container] + command

        if kwargs is not None:
            for key, value in kwargs.iteritems():
                print key, value
                if value == True:
                    docker_command += ['-' + str(key)]
                elif value == False:
                    continue
                else:
                    docker_command += ['-' + str(key) + ' ' + value]

    print ' '.join(docker_command)
    call(' '.join(docker_command), shell=True)