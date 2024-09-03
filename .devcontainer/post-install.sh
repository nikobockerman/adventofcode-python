#!/bin/bash

set -ex

WORKSPACE_DIR=$(pwd)

# Change some Poetry settings to make it more friendly in a container
poetry config cache-dir ${WORKSPACE_DIR}/.poetry_cache
poetry config virtualenvs.in-project true

# Now install all dependencies, including dev dependencies
poetry install --with=dev --sync

echo "Done!"
