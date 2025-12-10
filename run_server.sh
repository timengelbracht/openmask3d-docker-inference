#!/usr/bin/env bash
set -e

# Forward port 5001, enable GPU, run app.py
docker run --rm --gpus all -p 5001:5001 craiden/openmask:v1.0 python3 app.py
