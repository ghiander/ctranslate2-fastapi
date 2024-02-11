#!/bin/sh
# This script injects a model's files
# into the ct2-wrapper image without
# the need to rebuild when a new model
# is selected for inference.
#
# The ct2-wrapper base image must be
# built first using the Dockerfile from
# the base directory of ctranslate2-fastapi.
set -e  # exit on any error

MODEL_NAME="MBZUAI/LaMini-Flan-T5-248M"
docker create --name ct2-model ct2-wrapper
docker cp -a ${MODEL_NAME} ct2-model:/artifacts
docker commit ct2-model ct2-model
docker inspect ct2-model
echo ${MODEL_NAME} was injected into the image
