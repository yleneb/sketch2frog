from setuptools import setup, find_packages

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
)

# from the working directory (where this file is)
# pip install -e .
# installs the package "src" in editable mode
# so we can edit it without having to reinstall