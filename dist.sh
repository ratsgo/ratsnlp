#!/usr/bin/env bash

VERSION=$1

python setup.py sdist bdist_wheel
python -m twine upload dist/*$VERSION*