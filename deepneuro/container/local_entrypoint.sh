#!/bin/bash

cd /home/DeepNeuro
python3 /home/DeepNeuro/setup.py develop

exec "$@"