#!/bin/bash

set -ex

# Install python tools to be available outside venv
uv tool install poethepoet

# Synchronize venv and dependencies
uv sync

echo "Done!"
